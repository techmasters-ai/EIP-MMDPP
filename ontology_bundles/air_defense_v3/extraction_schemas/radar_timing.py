"""radar_timing extraction pass — pulse + scan timing.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Group fields: system_name, nominal_pri_usec,
nominal_pd_usec, scan_period_sec, dwell_time.

Field descriptions are sanitized at copy time per spec §4.4: numeric
fields reference DELTA_SYSTEM_PROMPT's Unit Policy block instead of
inlining conversion rules. The byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was
loosened in commit 20b1a8d to allow this divergence.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_timing"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarTimingRecord(BaseModel):
    """Subset of RadarSystemEntity covering pulse + scan timing."""

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
    nominal_pri_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Repetition Interval in microseconds. Emit "
            "only when the source states value AND unit. See Unit Policy "
            "in DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    nominal_pd_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Duration in microseconds. Emit only when the "
            "source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    scan_period_sec: Optional[float] = Field(
        default=None,
        description=(
            "Time to complete one full scan in seconds. Emit only when "
            "the source states value AND unit."
        ),
    )
    dwell_time: Optional[str] = Field(
        default=None,
        description=(
            "Time spent at a single beam position. Free-text; emit "
            "verbatim from the source."
        ),
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_nominal_pri_usec = field_validator("nominal_pri_usec", mode="before")(coerce_optional_float)
    _v_nominal_pd_usec  = field_validator("nominal_pd_usec", mode="before")(coerce_optional_float)
    _v_scan_period_sec  = field_validator("scan_period_sec", mode="before")(coerce_optional_float)
    _v_dwell_time       = field_validator("dwell_time", mode="before")(coerce_optional_text)


class RadarTimingPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarTimingRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with pulse + scan timing values "
            "extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={"dwell_time"},  # dwell_time is the one text field beyond identity
        )
    )
