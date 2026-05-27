"""radar_power_rf extraction pass — RF carrier + transmit power.

Spec §4.4. Group fields: system_name, erp_dbw, tx_peak_power_kw,
nominal_rf_mhz.

Field descriptions are sanitized at copy time per spec §4.4: numeric
fields reference DELTA_SYSTEM_PROMPT's Unit Policy block instead of
inlining cross-unit conversion rules. The byte-equal description-
parity check in test_extraction_views_subset_of_canonical_with_validator_parity
was loosened in commit 20b1a8d to allow this divergence; type /
validator / graph_id_fields parity remain enforced.
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
    """Subset of RadarSystemEntity covering RF carrier + transmit power."""

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
            "Effective Radiated Power in dBW. Source labels such as "
            "'ERP' or 'Effective Radiated Power' map here only when the "
            "source unit is dBW/dBm. Emit only when the source states "
            "the value with units; otherwise null. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    tx_peak_power_kw: Optional[float] = Field(
        default=None,
        description=(
            "Transmitter peak power in kilowatts. Source labels such as "
            "'Peak Power', 'Transmitter Power', 'Tx Power', or 'Pulse "
            "Power' map here when they describe peak transmitter power. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    nominal_rf_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Nominal carrier frequency in MHz. Source labels such as "
            "'Frequency', 'Operating Frequency', 'Carrier Frequency', "
            "or 'RF' map here when they describe the radar carrier. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
        ),
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
