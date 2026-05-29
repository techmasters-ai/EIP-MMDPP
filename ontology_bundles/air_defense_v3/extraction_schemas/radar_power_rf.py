"""radar_power_rf extraction pass — RF carrier + transmit power.

Spec §4.4. Group fields: system_name, erp_dbw, tx_peak_power_kw,
nominal_rf_mhz.

Field descriptions are written as dual-use retrieval and prompt text:
use document-facing anchor terms while preserving extraction constraints. The
byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was loosened
in commit 20b1a8d to allow this divergence; type / validator /
graph_id_fields parity remain enforced.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_power_rf"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]


class RadarPowerRfRecord(BaseModel):
    """Radar RF power and carrier-frequency characteristics: ERP, peak transmitter power, operating frequency, and radar band."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    erp_dbw: Optional[float] = Field(
        default=None,
        description=(
            "Effective Radiated Power in dBW. Relevant source labels include "
            "'ERP', 'EIRP', 'Effective Radiated Power', 'effective radiated "
            "power', 'radiated power', 'effective power', 'dBW', or 'dBm'. "
            "Use only effective radiated power or EIRP values stated with "
            "dBW/dBm units; otherwise null."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "ERP", "EIRP", "effective radiated power",
                    "effective isotropic radiated power",
                    "radiated power", "effective power",
                ],
                "negative_terms": [
                    "peak power", "transmitter power", "average power",
                    "generator power", "pulse power",
                ],
                "evidence_patterns": [
                    "ERP", "EIRP",
                    r"re:e(?:ffective)?\s+radiated\s+power",
                ],
                "likely_sections": [
                    "specifications", "performance", "transmitter",
                    "electromagnetic", "radar",
                ],
                "units": ["dBW", "dBm", "W", "kW"],
            }
        },
    )
    tx_peak_power_kw: Optional[float] = Field(
        default=None,
        description=(
            "Transmitter peak power in kilowatts. Relevant source labels include "
            "'Peak Power', 'peak transmitter power', 'Transmitter Power', "
            "'transmitter output power', 'Tx Power', 'Pulse Power', "
            "'peak pulse power', 'magnetron output', or 'klystron output'. "
            "Use peak transmitter or pulse output power, not ERP/EIRP or "
            "average power."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "peak power", "peak transmitter power", "transmitter power",
                    "Tx power", "TX power", "transmit power",
                    "pulse power", "peak pulse power",
                    "magnetron output", "klystron output",
                    "transmitter output power",
                ],
                "negative_terms": [
                    "effective radiated power", "ERP", "EIRP",
                    "average power", "generator power",
                ],
                "evidence_patterns": [
                    r"re:peak\s+(?:transmitter\s+)?power",
                    r"re:TX\s*power",
                    "magnetron output", "klystron output",
                ],
                "likely_sections": [
                    "specifications", "transmitter", "performance", "radar",
                ],
                "units": ["kW", "W", "MW", "dBW"],
            }
        },
    )
    nominal_rf_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Nominal RF carrier or operating frequency in MHz. Relevant source "
            "labels include 'Frequency', 'Operating Frequency', 'Carrier "
            "Frequency', 'RF', 'frequency range', 'waveband', 'radar band', "
            "'MHz', 'GHz', 'VHF', 'UHF', 'L-band', 'S-band', 'C-band', "
            "'X-band', or 'Ku-band'. Use radar carrier frequency, not "
            "PRF/PRI or modulation bandwidth."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "operating frequency", "carrier frequency", "RF frequency",
                    "frequency range", "waveband", "radar band",
                    "L-band", "S-band", "C-band", "X-band", "Ku-band",
                    "VHF", "UHF",
                ],
                "negative_terms": [
                    "PRF", "pulse repetition frequency",
                    "modulation bandwidth", "chirp bandwidth",
                    "doppler frequency",
                ],
                "evidence_patterns": [
                    r"re:\d+(?:\.\d+)?\s*(?:MHz|GHz)",
                    "operating frequency", "carrier frequency",
                ],
                "likely_sections": [
                    "specifications", "performance", "radar", "frequency",
                    "waveband",
                ],
                "units": ["MHz", "GHz", "kHz"],
            }
        },
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_erp_dbw          = field_validator("erp_dbw", mode="before")(coerce_optional_float)
    _v_tx_peak_power_kw = field_validator("tx_peak_power_kw", mode="before")(coerce_optional_float)
    _v_nominal_rf_mhz   = field_validator("nominal_rf_mhz", mode="before")(coerce_optional_float)


class RadarPowerRfPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarPowerRfRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with RF carrier + transmit power "
            "values extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields=set(),
        )
    )
