"""**LEGACY** — Not in the active manifest as of the
2026-04-27 radar field-group refactor. Replaced by:

- radar_identity.py
- radar_power_rf.py
- radar_antenna.py
- radar_timing.py
- radar_modulation.py

This module is kept in source as a reference for description text and
for legacy-loadability tests (e.g. test_service_identity_gate.py).
Do not add manifest entries pointing here. Will be removed in a
future cleanup once the new structure has been operationally proven.

---

Radar domain pass — flat, checklist-aligned schema.

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
    canonicalize_identity_text,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_text,
    coerce_optional_confidence,
    dedupe_entities_by_identity,
    sanitize_entity_list,
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

_RADAR_FORBIDDEN_SYSTEM_NAMES = {
    "SA-2", "SA-3", "SA-5", "SA-6", "SA-10", "SA-12", "SA-15", "SA-17",
    "SA-20", "SA-21", "SA-22", "SA-23", "PATRIOT", "PAC-2", "PAC-3",
    "PAC-3 MSE", "HAWK", "NIKE-HERCULES", "S-75", "S-125", "S-200", "S-300",
    "S-350", "S-400", "S-500", "AEGIS BMD", "SM-2", "SM-3", "SM-6", "THAAD",
    "ARROW", "IRON DOME", "DAVID'S SLING", "U-2", "SR-71", "RF-4C", "F-4",
    "F-15", "F-16", "B-52", "MIG-21", "MIG-23", "MIG-29", "SU-27",
}
_RADAR_OPTIONAL_TEXT_FIELDS = {
    "nomenclature",
    "elnot",
    "dieqp",
    "emitter_function",
    "system_status",
    "asrd",
    "responsible_agency",
    "review_cycle",
    "next_review_date",
    "scan_type",
    "intra_pulse_mop",
    "inter_pulse",
    "dwell_time",
}

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
            "FORBIDDEN values — never emit any of these as system_name "
            "because they are weapon/missile systems, not radars: "
            "SA-2, SA-3, SA-5, SA-6, SA-10, SA-12, SA-15, SA-17, SA-20, "
            "SA-21, SA-22, SA-23, Patriot, PAC-2, PAC-3, PAC-3 MSE, "
            "Hawk, Nike-Hercules, S-75, S-125, S-200, S-300, S-350, "
            "S-400, S-500, Aegis BMD, SM-2, SM-3, SM-6, THAAD, Arrow, "
            "Iron Dome, David's Sling. "
            "Also FORBIDDEN: aircraft / platform / target names (U-2, "
            "SR-71, RF-4C, F-4, F-15, F-16, B-52, MiG-21, MiG-23, "
            "MiG-29, Su-27) — these are targets that radars detect, "
            "not radars themselves. "
            "If the text says 'the SA-2 radar', emit the radar's own "
            "name ('Fan Song') if stated, otherwise omit. Do NOT emit "
            "'SA-2' here. "
            "Reject descriptive phrases ('the radar', 'the acquisition "
            "radar')."
        ),
        examples=["Fan Song", "Spoon Rest", "Tombstone", "AN/MPQ-65", "Flap Lid"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — the formal alphanumeric "
            "designation assigned by the manufacturing country. For US "
            "radars this is the JETDS / AN-style designator (e.g. "
            "'AN/MPQ-65'). For Russian / Soviet-origin radars it's the "
            "GRAU index or manufacturer model (e.g. '5N63S', '30N6E'). "
            "Distinct from system_name, which is the common (often NATO "
            "reporting) name. Emit when the document explicitly states "
            "the formal designation alongside the common name."
        ),
        examples=["AN/MPQ-65", "5N63S", "30N6E", "AN/SPY-1D"],
    )

    elnot: Optional[str] = Field(
        default=None,
        description=(
            "ELINT Notation (ELNOT) — an ELINT-community unique alphabetic "
            "code assigned to a specific emitter signal by signals "
            "intelligence databases (typically a 4- or 5-letter code). "
            "Only appears in intelligence-community source documents. "
            "Emit verbatim from the document — do not infer."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters (DIEQP) identifier — "
            "a cross-reference ID into the DIEQP database maintained by "
            "the MDE (Mission Data Engineering) community. Typically a "
            "short alphanumeric token. Only appears in IC / MDE source "
            "documents. Emit verbatim — do not infer."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the radar in an engagement kill-chain. "
            "Enum values and their meanings: "
            "SEARCH = early-warning / acquisition radar that detects "
            "targets at long range; "
            "TRACKING = radar that maintains target track after "
            "acquisition but does not provide the terminal weapon-guidance "
            "function; "
            "FIRE_CONTROL = terminal-guidance radar that provides the "
            "tracking signal used by the weapon system's seeker or "
            "command-guidance link. Guidance / illumination radars such as "
            "Fan Song belong here; "
            "MULTI_FUNCTION = a single radar that performs multiple roles "
            "(phased-array designs like AN/SPY-1 are typical examples); "
            "HEIGHT_FINDER = dedicated elevation-measurement radar paired "
            "with 2D search radars; "
            "NAV = navigation or weather radar (not a combat emitter)."
        ),
        examples=["SEARCH", "FIRE_CONTROL", "TRACKING", "MULTI_FUNCTION"],
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status of the radar system as described in the "
            "source. Typical values: OPERATIONAL (currently deployed), "
            "DEVELOPMENTAL (prototype or pre-IOC), RETIRED (withdrawn "
            "from service), UPGRADED (modified variant superseding the "
            "base model), EXPORTED (sold to foreign operators only). "
            "Emit only when the document explicitly states the status; do "
            "not infer OPERATIONAL from historical narrative or from the "
            "fact that the radar appears in a museum display."
        ),
        examples=["OPERATIONAL", "RETIRED", "DEVELOPMENTAL"],
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier — a catalog code from the All-Source "
            "Reference Document, a classified IC catalog of emitters. "
            "Emit verbatim when explicitly stated in the source; do not "
            "infer or cross-reference."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for maintaining the MDE record for "
            "this radar. Typically a 3-letter IC acronym (e.g. 'IWC' = "
            "Information Warfare Center, 'NASIC' = National Air and "
            "Space Intelligence Center, 'ONI' = Office of Naval "
            "Intelligence, 'NGIC' = National Ground Intelligence Center)."
        ),
        examples=["IWC", "NASIC", "ONI", "NGIC"],
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled cadence at which the MDE record for this radar is "
            "reviewed and re-validated. Typical values: 'annual', "
            "'biennial', '2-year', '3-year', or an explicit duration. "
            "Free-text; emit verbatim when stated."
        ),
        examples=["annual", "biennial", "3-year"],
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Date of the next scheduled MDE review. Prefer ISO 8601 "
            "(YYYY-MM-DD); otherwise emit the date string verbatim as "
            "written in the source."
        ),
        examples=["2026-06-30", "June 2026"],
    )

    # Power Characteristics
    erp_dbw: Optional[float] = Field(
        default=None,
        description=(
            "Effective Radiated Power in dBW (decibels relative to 1 watt). "
            "ERP = transmitter power × antenna gain, expressed on the log "
            "scale. Typical combat radars: 50-90 dBW. If the source gives "
            "ERP in dBm, subtract 30 (1 W = 30 dBm = 0 dBW). If given in "
            "watts or kilowatts, convert: dBW = 10 × log10(watts)."
        ),
        examples=[72.0, 85.0],
    )
    tx_peak_power_kw: Optional[float] = Field(
        default=None,
        description=(
            "Transmitter peak power output in kilowatts. This is the power "
            "at the transmitter's output port BEFORE the antenna — not "
            "effective radiated power. Typical S-band ground radars: "
            "100-1000 kW peak. If the source gives power in watts, divide "
            "by 1000; if in megawatts, multiply by 1000."
        ),
        examples=[150, 600, 1000],
    )
    gain_dbi: Optional[float] = Field(
        default=None,
        description=(
            "Peak antenna gain in dBi (decibels relative to an isotropic "
            "radiator). Higher gain = narrower main beam. Typical dish / "
            "parabolic radar antennas: 30-45 dBi. Phased arrays: 35-50 dBi."
        ),
        examples=[38.0, 42.0],
    )

    # Antenna Characteristics
    antenna_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether an antenna photograph is included in the record (Y/N). "
            "Use null when the document does not state this — do NOT "
            "default to false. Emit true only when the text explicitly "
            "indicates a photograph is included."
        ),
    )
    antenna_dim_az_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture width in the azimuth (horizontal) dimension, "
            "in meters. For a rectangular planar array this is the long "
            "dimension of the face; for a parabolic dish it's the "
            "horizontal diameter. Drives azimuth beamwidth via "
            "beamwidth ≈ (wavelength / aperture) × 51 degrees."
        ),
        examples=[4.5, 12.0],
    )
    antenna_dim_el_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture height in the elevation (vertical) dimension, "
            "in meters. Analogous to antenna_dim_az_m but for the "
            "elevation plane. A radar with asymmetric az/el dimensions has "
            "a fan beam (narrow azimuth, broad elevation) typical of 2D "
            "surveillance radars."
        ),
        examples=[2.5, 4.0],
    )
    beamwidth_az_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3dB azimuth beamwidth, in degrees. Defines the "
            "angular spread of the radar's beam in the horizontal plane "
            "between half-power points. Typical ground-based S-band / "
            "C-band radars: 1-4 degrees."
        ),
        examples=[1.5, 2.8],
    )
    beamwidth_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3dB elevation beamwidth, in degrees. The vertical "
            "analogue of beamwidth_az_deg. Fan-beam surveillance radars "
            "have intentionally wide elevation beamwidths (10-40°) to "
            "cover airspace with a single scan; pencil-beam tracking "
            "radars keep elevation beamwidth tight (1-4°)."
        ),
        examples=[1.5, 15.0],
    )
    spoiled: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the beam is spoiled (Y/N). "
            "Use null when the document does not state this — do NOT "
            "default to false. Emit true only when the text explicitly "
            "indicates a spoiled beam; emit false only when the text "
            "explicitly indicates an unspoiled beam."
        ),
    )
    coverage_limits_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum elevation angle (degrees) the radar can scan or "
            "track to. Ground-based search radars typically cap at "
            "30-40°; fire-control radars can go to 80°+. When the source "
            "gives a range ('0-45°'), emit the upper limit. Null when "
            "the document doesn't state a cap."
        ),
        examples=[45.0, 80.0],
    )

    # Parametric Characteristics
    nominal_rf_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Nominal operating RF (carrier frequency) in megahertz. "
            "Common radar bands: L-band 1000-2000, S-band 2000-4000, "
            "C-band 4000-8000, X-band 8000-12000, Ku-band 12000-18000. "
            "If the source gives GHz, multiply by 1000. If it gives a "
            "range, emit the center frequency."
        ),
        examples=[3000, 9400, 16000],
    )
    nominal_pri_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Repetition Interval (PRI) in microseconds — "
            "the time from the start of one pulse to the start of the "
            "next. PRI = 1 / PRF. Determines unambiguous range: "
            "unambiguous_range_km = PRI_usec × 0.15. Typical long-range "
            "search radars: 3000-10000 µs. Fire-control radars: "
            "100-1000 µs. If the source gives PRF in Hz, convert: "
            "PRI_usec = 1_000_000 / PRF_Hz."
        ),
        examples=[1000, 5000],
    )
    nominal_pd_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Duration / pulse width (PD) in microseconds — "
            "the length of a single transmitted pulse. Short pulses (0.1-1 "
            "µs) give fine range resolution; long pulses (50-200 µs) give "
            "more energy on target. Compressed-pulse radars specify the "
            "pre-compression PD here (the long pulse before matched filtering)."
        ),
        examples=[0.5, 50.0, 200.0],
    )
    scan_type: Optional[str] = Field(
        default=None,
        description=(
            "How the radar's beam is mechanically or electronically steered. "
            "Typical values: "
            "CIRCULAR (continuous 360° mechanical rotation), "
            "SECTOR (back-and-forth sweep over a limited arc), "
            "RASTER (2D sweep covering an elevation stack), "
            "ELECTRONIC (phased-array beam steering, no moving parts), "
            "DWELL_AND_SWITCH (mechanical slew with pause at each beam "
            "position), "
            "HELICAL (continuous rotation with simultaneous elevation "
            "stepping). Emit as uppercase when possible."
        ),
        examples=["CIRCULAR", "ELECTRONIC", "DWELL_AND_SWITCH"],
    )
    scan_period_sec: Optional[float] = Field(
        default=None,
        description=(
            "Time (seconds) to complete one full scan pattern — e.g. 360° "
            "rotation for a CIRCULAR scan, or one full raster for a "
            "RASTER scan. Combined with beamwidth this determines revisit "
            "rate. Typical rotating search radars: 4-12 s per revolution."
        ),
        examples=[4.0, 10.0],
    )

    # Modulation
    intra_pulse_mop: Optional[str] = Field(
        default=None,
        description=(
            "Intra-pulse modulation (Modulation On Pulse) — how the "
            "transmitted pulse is modulated for compression. "
            "Typical values: "
            "CW (continuous wave / unmodulated), "
            "LFM_CHIRP (linear frequency modulation / chirp), "
            "NLFM (non-linear FM), "
            "BARKER_CODE (binary phase-coded, Barker sequence), "
            "POLYPHASE (multi-level phase code, e.g. Frank / P1-P4), "
            "BIPHASE (2-level phase code). "
            "Free-text emission OK if the source uses different terms."
        ),
        examples=["LFM_CHIRP", "BARKER_CODE", "BIPHASE"],
    )
    frequency_excursion_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Frequency excursion (chirp bandwidth) in megahertz — for an "
            "LFM chirp waveform, the total swept bandwidth Δf across the "
            "pulse duration. Determines range resolution: "
            "resolution_m ≈ 150 / bandwidth_MHz. Only meaningful for "
            "frequency-modulated (chirped) waveforms."
        ),
        examples=[1.0, 10.0, 50.0],
    )
    num_bits_in_code: Optional[int] = Field(
        default=None,
        description=(
            "Number of chips in the phase-code sequence used for pulse "
            "compression. Only meaningful for phase-coded waveforms "
            "(Barker, polyphase, etc.). Common Barker codes: 7, 11, 13. "
            "Longer codes (128, 256, 1024) appear in modern systems. "
            "Null for CW or FM-chirp radars."
        ),
        examples=[7, 13, 1024],
    )
    inter_pulse: Optional[str] = Field(
        default=None,
        description=(
            "Inter-pulse modulation — how successive pulses vary. Typical "
            "patterns: "
            "CONSTANT_PRI (fixed spacing between pulses), "
            "PRI_STAGGER (multi-level PRI that cycles between values; "
            "resolves range ambiguity), "
            "PRI_JITTER (random PRI variation; anti-jam / ECCM), "
            "FREQ_AGILE (pulse-to-pulse frequency hopping). "
            "Emit as uppercase token when possible."
        ),
        examples=["CONSTANT_PRI", "PRI_STAGGER", "FREQ_AGILE"],
    )
    pulses_per_dwell: Optional[int] = Field(
        default=None,
        description=(
            "Number of pulses coherently or non-coherently integrated in "
            "one beam-position dwell. More pulses = more energy on target "
            "(and better Doppler resolution for coherent integration), at "
            "the cost of slower scan rate. Typical values: 8-64 for "
            "modern radars."
        ),
        examples=[16, 64],
    )
    dwell_time: Optional[str] = Field(
        default=None,
        description=(
            "Time spent at a single beam position (dwell-and-switch scans). "
            "Free-text because sources sometimes give a duration "
            "('12 ms'), a count ('16 pulses'), or a descriptive phrase. "
            "Only relevant for DWELL_AND_SWITCH or phased-array scan_types."
        ),
        examples=["12 ms", "16 pulses"],
    )

    # System
    confidence: Optional[float] = Field(
        default=None,
        description=(
            "Overall extraction confidence for this radar instance, "
            "0-1. Combines identity certainty + parametric confidence. "
            "Use 0.9-1.0 when identity is from a table/figure caption and "
            "parameters are explicit; 0.5-0.8 for prose mentions with "
            "partial parameters; <0.5 for inferred / reconstructed values. "
            "System-populated — leave null if unsure."
        ),
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    @field_validator("system_name", mode="before")
    @classmethod
    def _v_system_name(cls, value: Any) -> str:
        normalized = canonicalize_identity_text(value)
        if normalized is None:
            raise ValueError("system_name must be a non-empty radar identity from the document")
        return normalized

    _v_nomenclature             = field_validator("nomenclature",            mode="before")(coerce_optional_text)
    _v_elnot                    = field_validator("elnot",                   mode="before")(coerce_optional_text)
    _v_dieqp                    = field_validator("dieqp",                   mode="before")(coerce_optional_text)
    _v_emitter_function         = field_validator("emitter_function",        mode="before")(coerce_optional_text)
    _v_system_status            = field_validator("system_status",           mode="before")(coerce_optional_text)
    _v_asrd                     = field_validator("asrd",                    mode="before")(coerce_optional_text)
    _v_responsible_agency       = field_validator("responsible_agency",      mode="before")(coerce_optional_text)
    _v_review_cycle             = field_validator("review_cycle",            mode="before")(coerce_optional_text)
    _v_next_review_date         = field_validator("next_review_date",        mode="before")(coerce_optional_text)
    _v_scan_type                = field_validator("scan_type",               mode="before")(coerce_optional_text)
    _v_intra_pulse_mop          = field_validator("intra_pulse_mop",         mode="before")(coerce_optional_text)
    _v_inter_pulse              = field_validator("inter_pulse",             mode="before")(coerce_optional_text)
    _v_dwell_time               = field_validator("dwell_time",              mode="before")(coerce_optional_text)
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
            "prose using the radar's canonical designation (e.g. "
            "'Tombstone', 'AN/MPQ-65', 'Fan Song', 'Spoon Rest'). "
            "Do NOT emit missile-system names like 'SA-2' from "
            "unnamed descriptions ('the radar', 'the acquisition radar'). "
            "For mention-only evidence, emit identity plus directly "
            "stated properties; do not infer attachments not explicit in "
            "this batch."
        ),
        examples=[["Tombstone", "Fan Song"], ["AN/MPQ-65", "Spoon Rest"]],
        default_factory=list,
    )

    @model_validator(mode="before")
    @classmethod
    def _sanitize_and_dedupe_root_entities(cls, values: Any) -> Any:
        values = sanitize_entity_list(
            cls,
            values,
            list_field="radar_systems",
            identity_field="system_name",
            optional_text_fields=_RADAR_OPTIONAL_TEXT_FIELDS,
            forbidden_identities=_RADAR_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)
