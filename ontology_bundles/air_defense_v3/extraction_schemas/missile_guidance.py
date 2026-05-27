"""missile_guidance extraction pass — guidance + seeker + photo flag.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, guidance_type, seeker_type, missile_photo.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_guidance"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileGuidanceRecord(BaseModel):
    """Missile guidance and seeker characteristics: command guidance, homing mode, terminal seeker, and missile photo flag."""

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
            "canonical token from prose (e.g. '5V55K', '9M82', "
            "'AIM-120', 'PAC-3', 'SM-6'). When the document gives a "
            "formal designation PLUS a common/NATO name PLUS a "
            "block/variant (e.g. 'MIM-104F Patriot Advanced "
            "Capability-3 (PAC-3)'), emit ONLY the primary identifier "
            "('PAC-3' or 'MIM-104F'); DO NOT compress them all into "
            "system_name. CRITICAL: this name MUST match the "
            "system_name used by `missile_identity` for the same "
            "missile, so all field-group passes deduplicate onto one "
            "MISSILE_SYSTEM vertex during merge. Never emit radar, "
            "weapon-system, aircraft, or platform names — those are "
            "filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    guidance_type: Optional[str] = Field(
        default=None,
        description=(
            "Guidance scheme — how the missile is steered to the target. "
            "REQUIRED EMISSION: when the document contains any of "
            "these phrases (or close variants), emit the phrase "
            "verbatim from the source — these phrases ARE explicit "
            "guidance assignments: "
            "'command guidance' / 'CLOS' / 'command-to-line-of-sight'; "
            "'semi-active radar homing' / 'SARH'; "
            "'active radar homing' / 'ARH'; "
            "'track-via-missile' / 'TVM'; "
            "'inertial guidance' / 'inertial + mid-course update' / "
            "'inertial with mid-course'; "
            "'beam-rider' / 'beam riding'; "
            "'infrared homing' / 'IR homing'; "
            "'imaging infrared' / 'IIR'; "
            "'passive radar homing' / 'PRH'; "
            "'home-on-jam' / 'HOJ'. "
            "Free-text — preserve the source's exact phrasing. "
            "If multiple schemes are stated (e.g. 'SARH with terminal "
            "ARH'), emit them as a single string. Do NOT infer "
            "guidance from indirect cues."
        ),
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description=(
            "Seeker type — the missile's terminal-phase sensor. "
            "REQUIRED EMISSION: when the document contains any of "
            "these phrases (or close variants), emit the phrase "
            "verbatim from the source: "
            "'semi-active radar homing' / 'SARH seeker'; "
            "'active radar homing' / 'ARH seeker'; "
            "'X-band active seeker'; 'Ka-band millimeter-wave seeker' / "
            "'mmW seeker'; "
            "'infrared (IR)' / 'IR seeker'; "
            "'imaging infrared (IIR)' / 'IIR seeker'; "
            "'dual-mode IR/RF' / 'dual-mode seeker'; "
            "'passive radar homing'; "
            "'electro-optical' / 'EO seeker'. "
            "Free-text — preserve the source's exact phrasing. Do NOT "
            "infer the seeker from guidance prose alone unless the "
            "document uses one of these seeker-specific phrases."
        ),
    )
    # Optional[bool] uses Pydantic's native bool parsing — same pattern
    # as radar_antenna's antenna_photo / spoiled fields.
    missile_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a missile photograph or image is included in the "
            "record. Relevant source labels include 'photo', 'photograph', "
            "'image', 'figure', 'missile photo', or 'weapon photo'. "
            "Use null when not stated."
        ),
    )

    _v_system_name    = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_guidance_type  = field_validator("guidance_type", mode="before")(coerce_optional_text)
    _v_seeker_type    = field_validator("seeker_type", mode="before")(coerce_optional_text)
    # missile_photo: no validator — Pydantic native bool parsing.


class MissileGuidancePass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileGuidanceRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with guidance + seeker values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={"guidance_type", "seeker_type"},
        )
    )
