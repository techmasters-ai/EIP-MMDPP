"""radar_identity extraction pass — radar identity + administrative metadata.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Emits RADAR_SYSTEM[] with system_name as
the merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, elnot, dieqp, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date,
scan_type.

Field descriptions are sanitized at copy time per spec §4.4 (FORBIDDEN-
values block stripped from system_name; "typical X" enumeration prose
dropped) so the LLM-facing schema stays focused. Forbidden-system
filtering still happens — deterministically, via make_root_sanitizer.
The byte-equal description-parity check in
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
    """Subset of RadarSystemEntity covering identity + admin fields."""

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
            "radar names from prose (e.g. 'Fan Song', 'Spoon Rest', "
            "'Tombstone', 'AN/MPQ-65'). Never emit weapon, missile, "
            "aircraft, or platform names — those are filtered "
            "deterministically by the root sanitizer."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — formal alphanumeric "
            "designation (JETDS / AN-style for US, GRAU index for "
            "Russian/Soviet). Distinct from system_name."
        ),
    )
    elnot: Optional[str] = Field(
        default=None,
        description=(
            "ELINT Notation — community-unique alphabetic code from "
            "intelligence databases. Emit verbatim; do not infer."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters identifier. "
            "Emit verbatim; do not infer."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the radar in an engagement kill-chain. "
            "Accept one of: SEARCH, TRACKING, FIRE_CONTROL, "
            "MULTI_FUNCTION, HEIGHT_FINDER, NAV. Emit only when the "
            "document explicitly assigns the role."
        ),
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status. Accept one of: OPERATIONAL, DEVELOPMENTAL, "
            "RETIRED, UPGRADED, EXPORTED. Emit only when the document "
            "states it."
        ),
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier from the All-Source Reference Document. "
            "Emit verbatim when stated."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for the MDE record. 3-letter IC "
            "acronym (IWC, NASIC, ONI, NGIC)."
        ),
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled review cadence. Free-text; emit verbatim."
        ),
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Next scheduled MDE review date. ISO 8601 preferred."
        ),
    )
    scan_type: Optional[str] = Field(
        default=None,
        description=(
            "How the beam is steered. Accept one of: CIRCULAR, SECTOR, "
            "RASTER, ELECTRONIC, DWELL_AND_SWITCH, HELICAL. Emit as "
            "uppercase."
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
            "metadata extracted from this batch."
        ),
        examples=[["Fan Song", "Spoon Rest"]],
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
