"""radar_timing extraction pass — pulse + scan timing.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Group fields: system_name, nominal_pri_usec,
nominal_pd_usec, scan_period_sec, dwell_time.

Field descriptions are written as dual-use retrieval and prompt text:
use document-facing anchor terms while preserving extraction constraints. The
byte-equal description-parity check in
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
    """Radar pulse and scan timing: PRI, PRF interval, pulse width, scan period, rotation time, and dwell time."""

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
            "Nominal Pulse Repetition Interval in microseconds. Relevant "
            "source labels include 'PRI', 'Pulse Repetition Interval', "
            "'Pulse Interval', 'interpulse period', 'pulse spacing', "
            "'repetition interval', or 'microseconds'. Use interval time, "
            "not pulse repetition frequency in Hz unless the source gives "
            "or implies the interval."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "PRI", "pulse repetition interval",
                    "pulse interval", "interpulse period",
                    "pulse spacing", "repetition interval",
                ],
                "negative_terms": [
                    "PRF", "pulse repetition frequency",
                    "pulse width", "pulse duration", "scan period",
                ],
                "evidence_patterns": [
                    r"re:PRI",
                    r"re:pulse\s+repetition\s+interval",
                    r"re:interpulse\s+period",
                ],
                "likely_sections": [
                    "specifications", "timing", "waveform", "radar",
                    "performance",
                ],
                "units": ["µs", "us", "μs", "ms"],
            }
        },
    )
    nominal_pd_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal pulse duration or pulse width in microseconds. Relevant "
            "source labels include 'Pulse Width', 'Pulse Duration', "
            "'Pulse Length', 'PW', 'pulsewidth', 'transmitted pulse "
            "duration', or 'microseconds'. Use pulse width, not PRI/PRF."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "pulse width", "pulse duration", "pulse length",
                    "PW", "pulsewidth", "transmitted pulse duration",
                    "pulse time",
                ],
                "negative_terms": [
                    "PRI", "pulse repetition interval",
                    "PRF", "pulse repetition frequency",
                    "scan period", "dwell time",
                ],
                "evidence_patterns": [
                    r"re:PW\b",
                    r"re:pulse\s+(?:width|duration|length)",
                ],
                "likely_sections": [
                    "specifications", "timing", "waveform", "radar",
                    "transmitter",
                ],
                "units": ["µs", "us", "μs", "ns"],
            }
        },
    )
    scan_period_sec: Optional[float] = Field(
        default=None,
        description=(
            "Time to complete one full scan in seconds. Relevant source "
            "labels include 'Scan Period', 'Scan Time', 'Rotation Period', "
            "'antenna rotation time', 'scan cycle', 'revisit period', "
            "'rpm', or 'seconds per revolution'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "scan period", "scan time", "rotation period",
                    "antenna rotation time", "scan cycle",
                    "revisit period", "antenna rotation rate",
                    "revolution time",
                ],
                "negative_terms": [
                    "pulse width", "PRI", "dwell time",
                    "pulse repetition interval",
                ],
                "evidence_patterns": [
                    r"re:scan\s+(?:period|time|rate)",
                    r"re:rotation\s+(?:period|time|rate)",
                    r"re:\d+\s*rpm",
                ],
                "likely_sections": [
                    "antenna", "scan", "specifications", "performance", "radar",
                ],
                "units": ["s", "sec", "rpm"],
            }
        },
    )
    dwell_time: Optional[str] = Field(
        default=None,
        description=(
            "Time spent at a single beam position. Relevant source labels "
            "include 'dwell time', 'beam dwell', 'look time', 'time on "
            "target', 'illumination time', or 'track dwell'. Free-text; "
            "preserve source units such as ms, seconds, microseconds, or "
            "µs and emit verbatim when stated explicitly."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "dwell time", "beam dwell", "look time",
                    "time on target", "illumination time",
                    "track dwell", "beam dwell time",
                ],
                "negative_terms": [
                    "scan period", "rotation period",
                    "PRI", "pulse width",
                ],
                "evidence_patterns": [
                    r"re:dwell\s+time",
                    r"re:beam\s+dwell",
                    "time on target", "illumination time",
                ],
                "likely_sections": [
                    "antenna", "timing", "specifications", "radar",
                    "tracking",
                ],
                "units": ["ms", "µs", "us", "μs", "s"],
            }
        },
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
