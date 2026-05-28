"""radar_identity extraction pass — radar identity + administrative metadata.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Emits RADAR_SYSTEM[] with system_name as
the merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, elnot, dieqp, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date,
scan_type.

Field descriptions are written as dual-use retrieval and prompt text:
use document-facing anchor terms while preserving extraction constraints.
Forbidden-system filtering still happens deterministically via
make_root_sanitizer. The byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was
deliberately loosened to allow this divergence; type / validator /
graph_id_fields parity remain enforced.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_identity"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarIdentityRecord(BaseModel):
    """Radar identity and administrative metadata: common radar name, NATO reporting name, formal designation, ELNOT, DIEQP, role, status, agency, review dates, and scan type."""

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
            "Canonical designation of the RADAR — the SHORTEST canonical "
            "token from prose (e.g. 'Fan Song', 'Spoon Rest', "
            "'Tombstone', 'AN/MPQ-65'). When the document gives both a "
            "formal designation AND a common/NATO reporting name, put ONLY "
            "the primary radar identifier here; move the formal alphanumeric "
            "designation to `nomenclature` when distinct — DO NOT compress "
            "multiple aliases into system_name. Never emit weapon, missile, "
            "aircraft, or platform names — those are filtered "
            "deterministically by the root sanitizer. Reject generic phrases "
            "such as 'the radar' or 'the acquisition radar'."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature or formal alphanumeric "
            "designation. Relevant source labels include 'nomenclature', "
            "'designation', 'military designation', 'model', 'type', "
            "'variant', 'JETDS', 'AN/', 'GRAU index', '5N', or '30N'. "
            "Keep distinct from system_name."
        ),
    )
    elnot: Optional[str] = Field(
        default=None,
        description=(
            "ELINT Notation: community-unique alphabetic emitter code. "
            "Relevant source labels include 'ELNOT', 'ELINT notation', "
            "'emitter notation', 'signal code', or 'emitter code'. Emit "
            "verbatim; do not infer."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters identifier. "
            "Relevant source labels include 'DIEQP', 'Digital Intelligence "
            "Equipment Parameters', 'equipment parameter id', or "
            "'MDE identifier'. Emit verbatim; do not infer."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the radar in an engagement kill-chain. "
            "REQUIRED EMISSION: when the document contains any of "
            "these phrases (or close variants), emit the corresponding "
            "enum value — these phrases ARE explicit role assignments, "
            "not interpretation: "
            "'search radar' / 'early warning radar' / 'acquisition "
            "radar' / 'EW radar' → SEARCH; "
            "'tracking radar' → TRACKING; "
            "'fire-control radar' / 'engagement radar' / 'missile "
            "guidance radar' / 'illuminator' → FIRE_CONTROL; "
            "'multi-function radar' / 'multifunction' / 'MFR' / "
            "'AMDR' / 'air and missile defense radar' / 'air defense "
            "and surveillance radar' → MULTI_FUNCTION; "
            "'height finder' / 'height-finding radar' → HEIGHT_FINDER; "
            "'navigation radar' / 'marine navigation' → NAV. "
            "If multiple roles appear (e.g. 'multi-function fire-"
            "control radar'), prefer the more specific one — here "
            "FIRE_CONTROL. "
            "Allowed enum values: SEARCH, TRACKING, FIRE_CONTROL, "
            "MULTI_FUNCTION, HEIGHT_FINDER, NAV. Do NOT invent a role "
            "from indirect cues."
        ),
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status. "
            "REQUIRED EMISSION: when the document contains any of "
            "these phrases (or close variants), emit the corresponding "
            "enum value — these phrases ARE explicit status assignments: "
            "'operational' / 'in service' / 'deployed' / 'fielded' / "
            "'active' / 'integrated into' / 'in operational use' / "
            "'in production' / 'currently fielded' / 'in active "
            "service' / 'employed by' / 'used by' (when paired with a "
            "current operator) → OPERATIONAL; "
            "'developmental' / 'under development' / 'prototype' / "
            "'in testing' / 'pre-production' → DEVELOPMENTAL; "
            "'retired' / 'decommissioned' / 'phased out' / 'no longer "
            "in service' / 'withdrawn from service' → RETIRED; "
            "'upgraded' / 'modernized' / 'block upgrade' / 'service "
            "life extension' → UPGRADED; "
            "'exported' / 'foreign sales' / 'export variant' / 'foreign "
            "military sales' / 'FMS' → EXPORTED. "
            "Allowed enum values: OPERATIONAL, DEVELOPMENTAL, RETIRED, "
            "UPGRADED, EXPORTED. Do NOT infer status from age, era, "
            "or other indirect cues."
        ),
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier from the All-Source Reference Document. "
            "Relevant source labels include 'ASRD', 'All-Source Reference "
            "Document', or 'source reference id'. Emit verbatim when "
            "stated."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for the MDE record. Relevant "
            "source labels include 'responsible agency', 'agency', "
            "'OPR', 'office of primary responsibility', 'custodian', "
            "or 3-letter IC acronyms such as IWC, NASIC, ONI, NGIC, or MSIC."
        ),
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled review cadence. Relevant source labels include "
            "'review cycle', 'review cadence', 'update cycle', "
            "'review interval', 'annual', 'biennial', or 'triennial'. "
            "Free-text; emit verbatim."
        ),
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Next scheduled MDE review date. Relevant source labels "
            "include 'next review date', 'review date', 'next update', "
            "'due date', or 'scheduled review'. ISO 8601 preferred."
        ),
    )
    scan_type: Optional[str] = Field(
        default=None,
        description=(
            "How the beam is steered. Accept one of: CIRCULAR, SECTOR, "
            "RASTER, ELECTRONIC, DWELL_AND_SWITCH, HELICAL. "
            "Map prose to enum: 'mechanical rotation' / 'rotating "
            "antenna' / '360-degree scan' / 'rotating dish' → "
            "CIRCULAR; 'sector scan' / 'limited-arc' → SECTOR; "
            "'raster scan' / 'TV-style scan' → RASTER; "
            "'electronically scanned' / 'phased-array' / 'phased "
            "array' / 'AESA' / 'PESA' / 'ESA' / 'active "
            "electronically scanned array' / 'passive electronically "
            "scanned array' → ELECTRONIC; 'dwell-and-switch' / "
            "'step scan' → DWELL_AND_SWITCH; 'helical scan' / "
            "'spiral scan' → HELICAL. Emit as uppercase."
        ),
    )

    _v_system_name        = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_nomenclature       = field_validator("nomenclature", mode="before")(coerce_optional_text)
    _v_elnot              = field_validator("elnot", mode="before")(coerce_optional_text)
    _v_dieqp              = field_validator("dieqp", mode="before")(coerce_optional_text)
    _v_emitter_function   = field_validator("emitter_function", mode="before")(coerce_optional_text)
    _v_system_status      = field_validator("system_status", mode="before")(coerce_optional_text)
    _v_asrd               = field_validator("asrd", mode="before")(coerce_optional_text)
    _v_responsible_agency = field_validator("responsible_agency", mode="before")(coerce_optional_text)
    _v_review_cycle       = field_validator("review_cycle", mode="before")(coerce_optional_text)
    _v_next_review_date   = field_validator("next_review_date", mode="before")(coerce_optional_text)
    _v_scan_type          = field_validator("scan_type", mode="before")(coerce_optional_text)


class RadarIdentityPass(BaseModel):
    """Pass-root template — wraps radar_systems list."""

    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarIdentityRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with identity + administrative "
            "metadata extracted from this batch. If the batch contains no "
            "named radar systems, return an empty list."
        ),
        examples=[
            [{"system_name": "Fan Song"}, {"system_name": "Spoon Rest"}],
        ],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={
                "nomenclature", "elnot", "dieqp", "emitter_function",
                "system_status", "asrd", "responsible_agency",
                "review_cycle", "next_review_date", "scan_type",
            },
        )
    )
