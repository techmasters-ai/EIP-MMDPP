"""missile_identity extraction pass — missile identity + administrative metadata.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain into
smaller LLM call boundaries. Emits MISSILE_SYSTEM[] with system_name as
the merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, dieqp, name, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date.

Field descriptions are sanitized at copy time per spec §4.4 (FORBIDDEN-
values block stripped from system_name; "typical X" enumeration prose
dropped). The byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was
loosened in commit 20b1a8d (radar Task 4 unblock) to allow this.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_identity"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileIdentityRecord(BaseModel):
    """Subset of MissileSystemEntity covering identity + admin fields."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE — the SHORTEST "
            "canonical token from prose (e.g. '5V55K', '9M82', 'AIM-120', "
            "'MIM-104', 'RIM-174'). When the document gives a formal "
            "designation PLUS a common/NATO name PLUS a block/variant "
            "(e.g. 'MIM-104F Patriot Advanced Capability-3 (PAC-3)'), "
            "put ONLY the primary identifier here ('MIM-104F' or "
            "'PAC-3'); move the formal designation to `nomenclature` "
            "and the common/NATO reporting name to `name` — DO NOT "
            "compress them all into system_name. "
            "Never emit radar, weapon-system, aircraft, or platform "
            "names — those are filtered deterministically by the root "
            "sanitizer."
        ),
        examples=["5V55K", "AIM-120"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — formal alphanumeric "
            "designation. Distinct from system_name."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters identifier. "
            "Emit verbatim; do not infer."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Common or NATO reporting name when distinct from system_name. "
            "Examples: 'Guideline' (SA-2 / S-75), 'Goa' (SA-3 / S-125), "
            "'Gainful' (SA-6 / 2K12), 'Gammon' (SA-5 / S-200), 'Grumble' "
            "(SA-10 / S-300), 'Patriot' (MIM-104), 'Standard' (SM-2/SM-6). "
            "Free-text; emit verbatim from the source. Emit only when the "
            "source explicitly states a distinct common/NATO name (do not "
            "synthesize one)."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the missile. Emit only when the "
            "document explicitly assigns the role."
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
            "current operator) / 'paired with' (when paired with an "
            "operational radar/platform) → OPERATIONAL; "
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

    _v_system_name        = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_nomenclature       = field_validator("nomenclature", mode="before")(coerce_optional_text)
    _v_dieqp              = field_validator("dieqp", mode="before")(coerce_optional_text)
    _v_name               = field_validator("name", mode="before")(coerce_optional_text)
    _v_emitter_function   = field_validator("emitter_function", mode="before")(coerce_optional_text)
    _v_system_status      = field_validator("system_status", mode="before")(coerce_optional_text)
    _v_asrd               = field_validator("asrd", mode="before")(coerce_optional_text)
    _v_responsible_agency = field_validator("responsible_agency", mode="before")(coerce_optional_text)
    _v_review_cycle       = field_validator("review_cycle", mode="before")(coerce_optional_text)
    _v_next_review_date   = field_validator("next_review_date", mode="before")(coerce_optional_text)


class MissileIdentityPass(BaseModel):
    """Pass-root template — wraps missile_systems list."""

    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileIdentityRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with identity + administrative "
            "metadata extracted from this batch."
        ),
        examples=[["5V55K", "9M82"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={
                "nomenclature", "dieqp", "name", "emitter_function",
                "system_status", "asrd", "responsible_agency",
                "review_cycle", "next_review_date",
            },
        )
    )
