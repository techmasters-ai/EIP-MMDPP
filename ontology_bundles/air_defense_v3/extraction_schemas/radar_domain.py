"""Radar domain pass — flat, checklist-aligned schema.

Scope aligned to the `(U) RADAR MDE Checklist.xlsx` deliverable: every
extracted field corresponds 1:1 to a Data Element on that checklist.
Subcomponent classes (AntennaEntity, ReceiverEntity, TransmitterEntity,
SPCEntity, FrequencyBandEntity, WaveformEntity) are intentionally
removed — the checklist treats all parameters as system-level properties
of a single RADAR_SYSTEM instance. Canonical entities.py retains the
broader taxonomy for other bundles / future expansion.

Key entity (ontology_name):
- ``RADAR_SYSTEM`` — flattened to carry all checklist fields.

Tier 1 prose-mention rules on the pass-root list + identity field:
an instance may be emitted from EITHER a defining structure OR an
explicit named mention in prose using the system's canonical
designation. Unnamed descriptions are rejected.
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
    """Helper: declare a typed entity-to-entity edge field.

    Per docs "Template Basics → Edge Helper Function → Required Definition":
    this function must be defined identically in every template. Retained
    here for the pass-root → RadarSystemEntity CONTAINS edge even though
    the flat schema has no other typed edges.
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


# ----------------------------------------------------------------------
# Radar system entity — flat, checklist-driven fields.
# ----------------------------------------------------------------------

class RadarSystemEntity(BaseModel):
    """Radar system — flat extraction view aligned to the RADAR MDE Checklist.

    Every field below maps to a Data Element on that checklist. Numeric
    fields carry their unit in the field name (e.g. ``tx_peak_power_kw``)
    so the extractor does not need to canonicalize units separately.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    # Identity / Production Information
    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR itself. "
            "Accept canonical proper-noun radar names from prose when "
            "unambiguous (e.g. 'Fan Song', 'Spoon Rest', 'Tombstone', "
            "'Flap Lid', 'AN/MPQ-65'). "
            "Do NOT put a weapon-system designation here — 'SA-2', "
            "'Patriot', 'S-400' are missile/weapon systems, not radars. "
            "If the text says 'the SA-2 radar', emit the radar's own "
            "name ('Fan Song') if stated, otherwise omit. "
            "Reject descriptive phrases ('the radar', 'the acquisition "
            "radar') and target/platform names (U-2, SR-71)."
        ),
        examples=["Fan Song", "Spoon Rest", "Tombstone", "AN/MPQ-65", "Flap Lid"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military AN/ or NATO reporting nomenclature.",
        examples=["AN/MPQ-65", "5N63S"],
    )

    elnot: Optional[str] = Field(
        default=None,
        description="ELINT Notation (ELNOT) identifier.",
    )
    dieqp: Optional[str] = Field(
        default=None,
        description="Digital Intelligence Equipment Parameters (DIEQP) identifier.",
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description="Emitter function — what the radar is used for "
                    "(e.g. SEARCH, FIRE_CONTROL, TRACKING, MULTI_FUNCTION).",
        examples=["SEARCH", "FIRE_CONTROL", "TRACKING"],
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
        description="Next scheduled review date for the MDE record (free-text date).",
    )

    # Power Characteristics
    erp_dbw: Optional[float] = Field(
        default=None,
        description="Effective Radiated Power (ERP) in dBW.",
        examples=[72.0],
    )
    tx_peak_power_kw: Optional[float] = Field(
        default=None,
        description="Transmitter peak power output in kilowatts.",
        examples=[150],
    )
    gain_dbi: Optional[float] = Field(
        default=None,
        description="Antenna gain in dBi.",
        examples=[38.0],
    )

    # Antenna Characteristics
    antenna_photo: Optional[bool] = Field(
        default=None,
        description="Whether an antenna photograph is included in the record (Y/N).",
    )
    antenna_dim_az_m: Optional[float] = Field(
        default=None,
        description="Antenna azimuth dimension in meters.",
    )
    antenna_dim_el_m: Optional[float] = Field(
        default=None,
        description="Antenna elevation dimension in meters.",
    )
    beamwidth_az_deg: Optional[float] = Field(
        default=None,
        description="Azimuth beamwidth in degrees.",
        examples=[1.5],
    )
    beamwidth_el_deg: Optional[float] = Field(
        default=None,
        description="Elevation beamwidth in degrees.",
        examples=[1.5],
    )
    spoiled: Optional[bool] = Field(
        default=None,
        description="Whether the beam is spoiled (Y/N).",
    )
    coverage_limits_el_deg: Optional[float] = Field(
        default=None,
        description="Elevation coverage limits in degrees.",
    )

    # Parametric Characteristics
    nominal_rf_mhz: Optional[float] = Field(
        default=None,
        description="Nominal operating RF in MHz.",
        examples=[9400],
    )
    nominal_pri_usec: Optional[float] = Field(
        default=None,
        description="Nominal Pulse Repetition Interval (PRI) in microseconds.",
    )
    nominal_pd_usec: Optional[float] = Field(
        default=None,
        description="Nominal Pulse Duration (PD) in microseconds.",
    )
    scan_type: Optional[str] = Field(
        default=None,
        description="Scan type (e.g. circular, sector, raster, electronic, dwell-and-switch).",
    )
    scan_period_sec: Optional[float] = Field(
        default=None,
        description="Scan period in seconds.",
    )

    # Modulation
    intra_pulse_mop: Optional[str] = Field(
        default=None,
        description="Intra-pulse modulation (Modulation On Pulse / MOP).",
        examples=["LFM chirp", "Phase-coded"],
    )
    frequency_excursion_mhz: Optional[float] = Field(
        default=None,
        description="Frequency excursion in MHz (e.g. chirp bandwidth).",
    )
    num_bits_in_code: Optional[int] = Field(
        default=None,
        description="Number of bits in the intra-pulse code (if phase-coded).",
    )
    inter_pulse: Optional[str] = Field(
        default=None,
        description="Inter-pulse modulation descriptor.",
    )
    pulses_per_dwell: Optional[int] = Field(
        default=None,
        description="Number of pulses per dwell.",
    )
    dwell_time: Optional[str] = Field(
        default=None,
        description="Dwell time (dwell-and-switch scans only). "
                    "Free-text since the checklist unit column is blank.",
    )

    # System
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0-1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_erp_dbw                  = field_validator("erp_dbw",                 mode="before")(coerce_optional_float)
    _v_tx_peak_power_kw         = field_validator("tx_peak_power_kw",        mode="before")(coerce_optional_float)
    _v_gain_dbi                 = field_validator("gain_dbi",                mode="before")(coerce_optional_float)
    _v_antenna_dim_az_m         = field_validator("antenna_dim_az_m",        mode="before")(coerce_optional_float)
    _v_antenna_dim_el_m         = field_validator("antenna_dim_el_m",        mode="before")(coerce_optional_float)
    _v_beamwidth_az_deg         = field_validator("beamwidth_az_deg",        mode="before")(coerce_optional_float)
    _v_beamwidth_el_deg         = field_validator("beamwidth_el_deg",        mode="before")(coerce_optional_float)
    _v_coverage_limits_el_deg   = field_validator("coverage_limits_el_deg",  mode="before")(coerce_optional_float)
    _v_nominal_rf_mhz           = field_validator("nominal_rf_mhz",          mode="before")(coerce_optional_float)
    _v_nominal_pri_usec         = field_validator("nominal_pri_usec",        mode="before")(coerce_optional_float)
    _v_nominal_pd_usec          = field_validator("nominal_pd_usec",         mode="before")(coerce_optional_float)
    _v_scan_period_sec          = field_validator("scan_period_sec",         mode="before")(coerce_optional_float)
    _v_frequency_excursion_mhz  = field_validator("frequency_excursion_mhz", mode="before")(coerce_optional_float)
    _v_num_bits_in_code         = field_validator("num_bits_in_code",        mode="before")(coerce_optional_int)
    _v_pulses_per_dwell         = field_validator("pulses_per_dwell",        mode="before")(coerce_optional_int)
    _v_confidence               = field_validator("confidence",              mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass root
# ----------------------------------------------------------------------

class RadarDomainPass(BaseModel):
    """Radar-domain pass root. Emits only ``RADAR_SYSTEM`` entities with
    flat, checklist-aligned properties. No nested subcomponent entities
    and no typed HAS_* edges — the checklist treats everything as
    system-level.

    is_entity=True per docling-graph-docs.md Template Basics -> Root
    Document Model. graph_id_fields=[] because the pass-root is a
    synthetic container.
    """
    model_config = ConfigDict(
        extra="ignore",
        is_entity=True,
        graph_id_fields=[],
    )

    radar_systems: List[RadarSystemEntity] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems extracted from this document. "
            "Emit when the batch contains EITHER (a) a defining structure "
            "(table, caption, labeled list, captioned figure) that "
            "identifies the system, OR (b) an explicit named mention in "
            "prose using the system's canonical designation (e.g. 'SA-2', "
            "'Tombstone', 'AN/MPQ-65', 'Fan Song'). Do NOT emit from "
            "unnamed descriptions ('the radar', 'the acquisition radar'). "
            "For mention-only evidence, emit identity plus directly "
            "stated properties; do not infer attachments not explicit in "
            "this batch."
        ),
        examples=[["Tombstone", "Fan Song"], ["AN/MPQ-65", "SA-2 radar"]],
        default_factory=list,
    )

    _dedupe_root_entities = model_validator(mode="before")(dedupe_entities_by_identity)
