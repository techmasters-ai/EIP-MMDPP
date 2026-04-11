"""Radar domain pass: radar systems and all sub-components."""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import (
    coerce_optional_float,
    coerce_optional_confidence,
    normalize_enum,
)


class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    nomenclature: Optional[str] = None
    radar_type: Optional[str] = None
    nominal_frequency: Optional[str] = None
    tx_peak_power: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class AntennaEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    antenna_type: Optional[str] = None
    gain_dbi: Optional[float] = None
    beamwidth_az_deg: Optional[float] = None
    beamwidth_el_deg: Optional[float] = None
    confidence: Optional[float] = None

    _v_gain_dbi = field_validator("gain_dbi", mode="before")(coerce_optional_float)
    _v_beamwidth_az_deg = field_validator("beamwidth_az_deg", mode="before")(coerce_optional_float)
    _v_beamwidth_el_deg = field_validator("beamwidth_el_deg", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ReceiverEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    noise_figure_db: Optional[float] = None
    minimum_discernible_signal_dbm: Optional[float] = None
    confidence: Optional[float] = None

    _v_noise_figure_db = field_validator("noise_figure_db", mode="before")(coerce_optional_float)
    _v_minimum_discernible_signal_dbm = field_validator("minimum_discernible_signal_dbm", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class TransmitterEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    peak_power_at_transmitter_kw: Optional[float] = None
    duty_cycle: Optional[float] = None
    confidence: Optional[float] = None

    _v_peak_power_at_transmitter_kw = field_validator("peak_power_at_transmitter_kw", mode="before")(coerce_optional_float)
    _v_duty_cycle = field_validator("duty_cycle", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SPCEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    matched_filter_detection_loss_db: Optional[float] = None
    filter_response_type: Optional[str] = None
    confidence: Optional[float] = None

    _v_matched_filter_detection_loss_db = field_validator("matched_filter_detection_loss_db", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FrequencyBandEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [band_name]
    band_name: Optional[str] = None
    designation: Optional[str] = None
    freq_min_mhz: Optional[float] = None
    freq_max_mhz: Optional[float] = None
    confidence: Optional[float] = None

    _v_freq_min_mhz = field_validator("freq_min_mhz", mode="before")(coerce_optional_float)
    _v_freq_max_mhz = field_validator("freq_max_mhz", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class WaveformEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [waveform_name]
    waveform_name: Optional[str] = None
    waveform_family: Optional[str] = None
    nominal_pulse_duration_us: Optional[float] = None
    nominal_PRI_us: Optional[float] = None
    duty_cycle: Optional[float] = None
    confidence: Optional[float] = None

    _v_nominal_pulse_duration_us = field_validator("nominal_pulse_duration_us", mode="before")(coerce_optional_float)
    _v_nominal_PRI_us = field_validator("nominal_PRI_us", mode="before")(coerce_optional_float)
    _v_duty_cycle = field_validator("duty_cycle", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PlatformEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    platform_type: Optional[str] = None
    service_branch: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SpecificationEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [parameter, value]
    parameter: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class RadarRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = None
    from_type: Optional[str] = None
    from_identity: Optional[dict[str, Any]] = None
    to_type: Optional[str] = None
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None

    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({
            "INSTALLED_ON", "HAS_ANTENNA", "HAS_RECEIVER", "HAS_TRANSMITTER",
            "HAS_PROCESSING_CHAIN", "OPERATES_IN_BAND", "USES_WAVEFORM", "SPECIFIED_BY",
        })
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class RadarDomainPass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    radar_systems: list[RadarSystemEntity] = Field(default_factory=list)
    antennas: list[AntennaEntity] = Field(default_factory=list)
    receivers: list[ReceiverEntity] = Field(default_factory=list)
    transmitters: list[TransmitterEntity] = Field(default_factory=list)
    signal_processing_chains: list[SPCEntity] = Field(default_factory=list)
    frequency_bands: list[FrequencyBandEntity] = Field(default_factory=list)
    waveforms: list[WaveformEntity] = Field(default_factory=list)
    platforms: list[PlatformEntity] = Field(default_factory=list)
    specifications: list[SpecificationEntity] = Field(default_factory=list)
    relationships: list[RadarRelationship] = Field(default_factory=list)
