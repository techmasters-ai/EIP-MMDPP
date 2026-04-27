"""radar_antenna extraction pass — antenna geometry + beam parameters.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Group fields: system_name, antenna_photo,
gain_dbi, antenna_dim_az_m, antenna_dim_el_m, beamwidth_az_deg,
beamwidth_el_deg, spoiled, coverage_limits_el_deg.

Field descriptions are sanitized at copy time per spec §4.4: numeric
fields reference DELTA_SYSTEM_PROMPT's Unit Policy block instead of
inlining conversion rules. The byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was
loosened in commit 20b1a8d to allow this divergence.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_antenna"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarAntennaRecord(BaseModel):
    """Subset of RadarSystemEntity covering antenna geometry + beam shape."""

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
    # Optional[bool] fields rely on Pydantic's native bool parsing; no
    # coerce_optional_bool helper exists in ..validators and we don't add
    # one in Session 1 (would be scope creep). Pydantic accepts JSON
    # true/false/null natively.
    antenna_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether an antenna photograph is included in the record. "
            "Use null when not stated."
        ),
    )
    gain_dbi: Optional[float] = Field(
        default=None,
        description=(
            "Peak antenna gain in dBi. Emit only when the source states "
            "value AND unit. See Unit Policy in DELTA_SYSTEM_PROMPT for "
            "dB-domain conversions."
        ),
    )
    antenna_dim_az_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture width (azimuth dimension) in meters. Emit "
            "only when the source states value AND unit."
        ),
    )
    antenna_dim_el_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture height (elevation dimension) in meters. "
            "Emit only when the source states value AND unit."
        ),
    )
    beamwidth_az_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3 dB azimuth beamwidth in degrees. Emit only when "
            "the source states value AND unit."
        ),
    )
    beamwidth_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3 dB elevation beamwidth in degrees. Emit only "
            "when the source states value AND unit."
        ),
    )
    spoiled: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the beam is spoiled (deliberately broadened). Use "
            "null when not stated."
        ),
    )
    coverage_limits_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum elevation coverage angle in degrees. Emit only when "
            "the source states value AND unit."
        ),
    )

    _v_system_name             = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_gain_dbi                = field_validator("gain_dbi", mode="before")(coerce_optional_float)
    _v_antenna_dim_az_m        = field_validator("antenna_dim_az_m", mode="before")(coerce_optional_float)
    _v_antenna_dim_el_m        = field_validator("antenna_dim_el_m", mode="before")(coerce_optional_float)
    _v_beamwidth_az_deg        = field_validator("beamwidth_az_deg", mode="before")(coerce_optional_float)
    _v_beamwidth_el_deg        = field_validator("beamwidth_el_deg", mode="before")(coerce_optional_float)
    _v_coverage_limits_el_deg  = field_validator("coverage_limits_el_deg", mode="before")(coerce_optional_float)
    # antenna_photo, spoiled: no validator — Pydantic native bool parsing.


class RadarAntennaPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarAntennaRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with antenna geometry + beam "
            "parameters extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields=set(),  # no text fields in this group beyond system_name
        )
    )
