"""Radar domain pass: radar systems and sub-components.

Narrow extraction views of canonical radar-chain entities. Parallel
classes with matching ``ontology_name`` — NOT subclasses — so pass
narrowing stays local while merge rejoins by identity.

Key entities (ontology_name):
- ``RADAR_SYSTEM`` — radar platform with RF + processing chain
- ``ANTENNA`` — antenna with polarization, beamwidth, gain, geometry
- ``RECEIVER`` — receiver with noise + sensitivity characteristics
- ``TRANSMITTER`` — transmitter with power + loss characteristics
- ``SIGNAL_PROCESSING_CHAIN`` — processing chain (detection, integration, Doppler)
- ``FREQUENCY_BAND`` — IEEE/NATO band designation
- ``WAVEFORM`` — radar waveform with pulse/timing characteristics
- ``PLATFORM`` — host platform
- ``SPECIFICATION`` — measurable parameter/value/unit/condition

Key relationships (docs-valid RelationshipType labels):
- ``RADAR_SYSTEM HAS_ANTENNA ANTENNA``
- ``RADAR_SYSTEM HAS_RECEIVER RECEIVER``
- ``RADAR_SYSTEM HAS_TRANSMITTER TRANSMITTER``
- ``RADAR_SYSTEM HAS_PROCESSING_CHAIN SIGNAL_PROCESSING_CHAIN``
- ``RADAR_SYSTEM OPERATES_IN_BAND FREQUENCY_BAND``
- ``RADAR_SYSTEM USES_WAVEFORM WAVEFORM``
- ``RADAR_SYSTEM INSTALLED_ON PLATFORM``

Plan v32 Task 46 (Phase 5 docs-compliance rewrite + typed-edge migration).
Intra-pass ``RadarRelationship`` DTO retained here until Task 64 deletes it.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import (
    coerce_optional_float,
    coerce_optional_confidence,
    coerce_optional_text,
)


def edge(label: str, **field_kwargs: Any) -> Any:
    """Helper: declare a typed entity-to-entity edge field.

    Per docs "Template Basics → Edge Helper Function → Required Definition":
    this function must be defined identically in every template.
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    return Field(json_schema_extra=existing_extra, **field_kwargs)


def _dedupe_entities_by_identity(cls, values):
    """Merge-preserving dedup for pass-root entity lists (plan Task 9d).

    For each entity list whose inner class carries non-empty
    ``graph_id_fields``, collapse duplicates by identity tuple and union
    non-null non-identity scalar fields (first-non-null wins).
    """
    if not isinstance(values, dict):
        return values
    for fname, finfo in cls.model_fields.items():
        raw = values.get(fname)
        if not isinstance(raw, list) or not raw:
            continue
        from typing import get_args as _ga
        def _find_basemodel(ann):
            if ann is None:
                return None
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                return ann
            for a in _ga(ann):
                r = _find_basemodel(a)
                if r is not None:
                    return r
            return None
        inner_cls = _find_basemodel(getattr(finfo, "annotation", None))
        if inner_cls is None:
            continue
        id_fields = list((inner_cls.model_config or {}).get("graph_id_fields", []) or [])
        if not id_fields:
            continue
        accum: dict[tuple, dict] = {}
        order: list[tuple] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = tuple(item.get(k) for k in id_fields)
            if key not in accum:
                accum[key] = dict(item)
                order.append(key)
            else:
                merged = accum[key]
                for k, v in item.items():
                    if k in id_fields or v is None:
                        continue
                    if k not in merged or merged[k] is None:
                        merged[k] = v
        if len(accum) != len(raw):
            values[fname] = [accum[k] for k in order]
    return values


# ----------------------------------------------------------------------
# Entities (narrow extraction views — subset of canonical entities.py)
# ----------------------------------------------------------------------

class AntennaEntity(BaseModel):
    """Antenna — narrow extraction view of canonical ANTENNA."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="ANTENNA",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Name or designation of the antenna",
        examples=["Main Array Antenna", "Main Array Antenna"],
    )
    antenna_type: Optional[str] = Field(
        default=None,
        description="Physical type of antenna",
        json_schema_extra={
            "enum": ["PHASED_ARRAY", "DISH", "DIPOLE", "HORN", "PATCH",
                     "YAGI", "CONFORMAL", "SLOTTED_ARRAY", "ESA"],
        },
    )
    gain_dbi: Optional[float] = Field(
        default=None, description="Peak antenna gain in dBi", examples=[38.0],
    )
    beamwidth_az_deg: Optional[float] = Field(
        default=None, description="Azimuth beamwidth in degrees", examples=[1.5],
    )
    beamwidth_el_deg: Optional[float] = Field(
        default=None, description="Elevation beamwidth in degrees", examples=[1.5],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_gain_dbi = field_validator("gain_dbi", mode="before")(coerce_optional_float)
    _v_beamwidth_az_deg = field_validator("beamwidth_az_deg", mode="before")(coerce_optional_float)
    _v_beamwidth_el_deg = field_validator("beamwidth_el_deg", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ReceiverEntity(BaseModel):
    """Receiver — narrow extraction view of canonical RECEIVER."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RECEIVER",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Name or designation of the receiver",
        examples=["Main Receiver Unit", "Main Receiver Unit"],
    )
    noise_figure_db: Optional[float] = Field(
        default=None, description="Receiver noise figure in dB", examples=[3.5],
    )
    minimum_discernible_signal_dbm: Optional[float] = Field(
        default=None,
        description="Minimum detectable signal level in dBm",
        examples=[-110],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_noise_figure_db = field_validator("noise_figure_db", mode="before")(coerce_optional_float)
    _v_minimum_discernible_signal_dbm = field_validator("minimum_discernible_signal_dbm", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class TransmitterEntity(BaseModel):
    """Transmitter — narrow extraction view of canonical TRANSMITTER."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="TRANSMITTER",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Name or designation of the transmitter",
        examples=["Main Transmitter Unit", "Main Transmitter Unit"],
    )
    peak_power_at_transmitter_kw: Optional[float] = Field(
        default=None,
        description="Peak power at the transmitter output in kilowatts",
        examples=[150],
    )
    duty_cycle: Optional[float] = Field(
        default=None,
        description="Transmitter duty cycle (0.0 to 1.0)",
        examples=[0.05],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_peak_power_at_transmitter_kw = field_validator("peak_power_at_transmitter_kw", mode="before")(coerce_optional_float)
    _v_duty_cycle = field_validator("duty_cycle", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SPCEntity(BaseModel):
    """Signal Processing Chain — narrow extraction view of canonical
    SIGNAL_PROCESSING_CHAIN.

    Class named SPCEntity (short form) but ontology_name matches canonical
    SignalProcessingChainEntity — merge rejoins by ontology_name, not class
    name. Plan Task 9a contract validates subset parity by ontology_name.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SIGNAL_PROCESSING_CHAIN",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Name of the signal processing chain",
        examples=["Main Processing Chain", "Main Processing Chain"],
    )
    matched_filter_detection_loss_db: Optional[float] = Field(
        default=None,
        description="Loss relative to ideal matched filter detection in dB",
        examples=[1.5],
    )
    filter_response_type: Optional[str] = Field(
        default=None,
        description="Doppler filter bank response type",
        examples=["FFT with Chebyshev window"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_matched_filter_detection_loss_db = field_validator("matched_filter_detection_loss_db", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FrequencyBandEntity(BaseModel):
    """Frequency Band — narrow extraction view of canonical FREQUENCY_BAND."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="FREQUENCY_BAND",
        graph_id_fields=["band_name"],
        identity_scope="global",
        is_entity=True,
    )

    band_name: str = Field(
        ...,
        description="Common name of the frequency band",
        examples=["X-band", "X-band"],
    )
    designation: Optional[str] = Field(
        default=None,
        description="IEEE or NATO band letter designation",
        json_schema_extra={
            "enum": ["HF", "VHF", "UHF", "L", "S", "C", "X", "Ku", "K",
                     "Ka", "V", "W", "mm"],
        },
    )
    freq_min_mhz: Optional[float] = Field(
        default=None,
        description="Lower frequency limit of the band in MHz",
        examples=[8000],
    )
    freq_max_mhz: Optional[float] = Field(
        default=None,
        description="Upper frequency limit of the band in MHz",
        examples=[12000],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_freq_min_mhz = field_validator("freq_min_mhz", mode="before")(coerce_optional_float)
    _v_freq_max_mhz = field_validator("freq_max_mhz", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class WaveformEntity(BaseModel):
    """Waveform — narrow extraction view of canonical WAVEFORM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="WAVEFORM",
        graph_id_fields=["waveform_name"],
        identity_scope="document",
        is_entity=True,
    )

    waveform_name: str = Field(
        ...,
        description="Name of the waveform mode",
        examples=["Search Mode 1", "Search Mode 1"],
    )
    waveform_family: Optional[str] = Field(
        default=None,
        description="Family or class of the waveform",
        json_schema_extra={
            "enum": ["PULSE_DOPPLER", "FMCW", "CW", "LFM_CHIRP",
                     "PHASE_CODED", "BURST", "PULSE"],
        },
    )
    nominal_pulse_duration_us: Optional[float] = Field(
        default=None,
        description="Nominal pulse duration in microseconds",
        examples=[10.0],
    )
    nominal_PRI_us: Optional[float] = Field(
        default=None,
        description="Nominal pulse repetition interval in microseconds",
        examples=[500],
    )
    duty_cycle: Optional[float] = Field(
        default=None,
        description="Waveform duty cycle ratio (0.0 to 1.0)",
        examples=[0.02],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_nominal_pulse_duration_us = field_validator("nominal_pulse_duration_us", mode="before")(coerce_optional_float)
    _v_nominal_PRI_us = field_validator("nominal_PRI_us", mode="before")(coerce_optional_float)
    _v_duty_cycle = field_validator("duty_cycle", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PlatformEntity(BaseModel):
    """Platform — narrow extraction view of canonical PLATFORM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="PLATFORM",
        graph_id_fields=["name"],
        identity_scope="global",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Common name of the platform",
        examples=["SA-20 TEL", "SA-20 TEL"],
    )
    platform_type: Optional[str] = Field(
        default=None,
        description="Category of the platform",
        json_schema_extra={
            "enum": ["AIRCRAFT", "SHIP", "GROUND_VEHICLE", "FIXED_SITE",
                     "SPACE", "MOBILE_LAUNCHER", "AIR_DEFENSE_BATTERY"],
        },
    )
    service_branch: Optional[str] = Field(
        default=None,
        description="Military branch operating the platform",
        examples=["Russian Aerospace Forces"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class RadarSystemEntity(BaseModel):
    """Radar System — narrow extraction view of canonical RADAR_SYSTEM
    with typed edges to the RF chain, processing, and host platform.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the radar system",
        examples=["Tombstone", "Tombstone"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military AN/ or NATO reporting nomenclature",
        examples=["AN/MPQ-65"],
    )
    radar_type: Optional[str] = Field(
        default=None,
        description="Functional category of the radar",
        json_schema_extra={
            "enum": ["SEARCH", "FIRE_CONTROL", "TRACKING", "WEATHER",
                     "SAR", "GMTI", "MULTI_FUNCTION", "AESA", "PESA", "MPAR"],
        },
    )
    nominal_frequency: Optional[str] = Field(
        default=None,
        description="Nominal operating frequency or center frequency",
        examples=["9.4 GHz"],
    )
    tx_peak_power: Optional[str] = Field(
        default=None,
        description="Transmitter peak power output",
        examples=["150 kW"],
    )

    # Typed edges to sub-components (HAS_*) and host platform (INSTALLED_ON).
    antennas: List[AntennaEntity] = edge(label="HAS_ANTENNA", default_factory=list)
    receivers: List[ReceiverEntity] = edge(label="HAS_RECEIVER", default_factory=list)
    transmitters: List[TransmitterEntity] = edge(label="HAS_TRANSMITTER", default_factory=list)
    signal_processing_chains: List[SPCEntity] = edge(label="HAS_PROCESSING_CHAIN", default_factory=list)
    frequency_bands: List[FrequencyBandEntity] = edge(label="OPERATES_IN_BAND", default_factory=list)
    waveforms: List[WaveformEntity] = edge(label="USES_WAVEFORM", default_factory=list)
    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SpecificationEntity(BaseModel):
    """Specification — narrow extraction view of canonical SPECIFICATION.

    Identity: ``parameter`` + ``value`` (document-scoped). Identity fields
    are required per plan docs-compliance; the coerce_optional_text
    pre-validator still normalises numeric payloads to strings (LLMs emit
    numeric parameter/value for table-row extractions).
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SPECIFICATION",
        graph_id_fields=["parameter", "value"],
        identity_scope="document",
        is_entity=True,
    )

    parameter: str = Field(
        ...,
        description="Name of the measured parameter (e.g. max_range, operating_temperature)",
        examples=["max_range", "max_range"],
    )
    value: str = Field(
        ...,
        description="Numeric value or range of the measurement",
        examples=["150", "150"],
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (SI or military standard)",
        examples=["km"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_parameter = field_validator("parameter", mode="before")(coerce_optional_text)
    _v_value = field_validator("value", mode="before")(coerce_optional_text)
    _v_unit = field_validator("unit", mode="before")(coerce_optional_text)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass-root container — is_entity=False (Decision 4a).
# Intra-pass RadarRelationship DTO removed in Task 64 (plan Task 37);
# typed edges on RadarSystemEntity replace it. SystemLinkRelationship in
# system_links.py remains the only surviving DTO (Decision 4).
# ----------------------------------------------------------------------

class RadarDomainPass(BaseModel):
    """Radar-domain pass root. Typed-edge migration: sub-components
    (antennas, receivers, transmitters, SPCs, freq bands, waveforms,
    platforms) are reached via typed edges on ``RadarSystemEntity``, not
    pass-root lists. ``specifications`` remains at pass root as it is a
    bridge entity not owned by any single radar.
    """
    model_config = ConfigDict(extra="ignore", is_entity=False)

    radar_systems: List[RadarSystemEntity] = Field(default_factory=list)
    specifications: List[SpecificationEntity] = Field(default_factory=list)

    _dedupe_root_entities = model_validator(mode="before")(_dedupe_entities_by_identity)
