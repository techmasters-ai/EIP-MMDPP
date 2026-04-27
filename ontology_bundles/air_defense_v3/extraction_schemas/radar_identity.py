"""radar_identity extraction pass — radar identity + administrative metadata.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Emits RADAR_SYSTEM[] with system_name as
the merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, elnot, dieqp, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date,
scan_type.

Field descriptions are copied verbatim from the canonical
``entities.RadarSystemEntity`` per the docs-compliance contract
(``test_extraction_views_subset_of_canonical_with_validator_parity``):
extraction-view descriptions must match the canonical byte-for-byte.
Forbidden-system filtering is enforced deterministically by the root
sanitizer (``make_root_sanitizer``); the canonical FORBIDDEN-values
prose is retained in the description so the LLM and the runtime
filter agree.
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
            "Canonical designation of the RADAR itself. "
            "Accept canonical proper-noun radar names from prose when "
            "unambiguous (e.g. 'Fan Song', 'Spoon Rest', 'Tombstone', "
            "'Flap Lid', 'AN/MPQ-65'). "
            "FORBIDDEN values — never emit any of these as system_name "
            "because they are weapon/missile systems, not radars: "
            "SA-2, SA-3, SA-5, SA-6, SA-10, SA-12, SA-15, SA-17, SA-20, "
            "SA-21, SA-22, SA-23, Patriot, PAC-2, PAC-3, PAC-3 MSE, "
            "Hawk, Nike-Hercules, S-75, S-125, S-200, S-300, S-350, "
            "S-400, S-500, Aegis BMD, SM-2, SM-3, SM-6, THAAD, Arrow, "
            "Iron Dome, David's Sling. "
            "Also FORBIDDEN: aircraft / platform / target names (U-2, "
            "SR-71, RF-4C, F-4, F-15, F-16, B-52, MiG-21, MiG-23, "
            "MiG-29, Su-27) — these are targets that radars detect, "
            "not radars themselves. "
            "If the text says 'the SA-2 radar', emit the radar's own "
            "name ('Fan Song') if stated, otherwise omit. Do NOT emit "
            "'SA-2' here. "
            "Reject descriptive phrases ('the radar', 'the acquisition "
            "radar')."
        ),
        examples=["Fan Song", "Spoon Rest", "Tombstone", "AN/MPQ-65", "Flap Lid"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — the formal alphanumeric "
            "designation assigned by the manufacturing country. For US "
            "radars this is the JETDS / AN-style designator (e.g. "
            "'AN/MPQ-65'). For Russian / Soviet-origin radars it's the "
            "GRAU index or manufacturer model (e.g. '5N63S', '30N6E'). "
            "Distinct from system_name, which is the common (often NATO "
            "reporting) name. Emit when the document explicitly states "
            "the formal designation alongside the common name."
        ),
        examples=["AN/MPQ-65", "5N63S", "30N6E", "AN/SPY-1D"],
    )
    elnot: Optional[str] = Field(
        default=None,
        description=(
            "ELINT Notation (ELNOT) — an ELINT-community unique alphabetic "
            "code assigned to a specific emitter signal by signals "
            "intelligence databases (typically a 4- or 5-letter code). "
            "Only appears in intelligence-community source documents. "
            "Emit verbatim from the document — do not infer."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters (DIEQP) identifier — "
            "a cross-reference ID into the DIEQP database maintained by "
            "the MDE (Mission Data Engineering) community. Typically a "
            "short alphanumeric token. Only appears in IC / MDE source "
            "documents. Emit verbatim — do not infer."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the radar in an engagement kill-chain. "
            "Enum values and their meanings: "
            "SEARCH = early-warning / acquisition radar that detects "
            "targets at long range; "
            "TRACKING = radar that maintains target track after "
            "acquisition but does not provide the terminal weapon-guidance "
            "function; "
            "FIRE_CONTROL = terminal-guidance radar that provides the "
            "tracking signal used by the weapon system's seeker or "
            "command-guidance link. Guidance / illumination radars such as "
            "Fan Song belong here; "
            "MULTI_FUNCTION = a single radar that performs multiple roles "
            "(phased-array designs like AN/SPY-1 are typical examples); "
            "HEIGHT_FINDER = dedicated elevation-measurement radar paired "
            "with 2D search radars; "
            "NAV = navigation or weather radar (not a combat emitter)."
        ),
        examples=["SEARCH", "FIRE_CONTROL", "TRACKING", "MULTI_FUNCTION"],
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status of the radar system as described in the "
            "source. Typical values: OPERATIONAL (currently deployed), "
            "DEVELOPMENTAL (prototype or pre-IOC), RETIRED (withdrawn "
            "from service), UPGRADED (modified variant superseding the "
            "base model), EXPORTED (sold to foreign operators only). "
            "Emit only when the document explicitly states the status; do "
            "not infer OPERATIONAL from historical narrative or from the "
            "fact that the radar appears in a museum display."
        ),
        examples=["OPERATIONAL", "RETIRED", "DEVELOPMENTAL"],
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier — a catalog code from the All-Source "
            "Reference Document, a classified IC catalog of emitters. "
            "Emit verbatim when explicitly stated in the source; do not "
            "infer or cross-reference."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for maintaining the MDE record for "
            "this radar. Typically a 3-letter IC acronym (e.g. 'IWC' = "
            "Information Warfare Center, 'NASIC' = National Air and "
            "Space Intelligence Center, 'ONI' = Office of Naval "
            "Intelligence, 'NGIC' = National Ground Intelligence Center)."
        ),
        examples=["IWC", "NASIC", "ONI", "NGIC"],
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled cadence at which the MDE record for this radar is "
            "reviewed and re-validated. Typical values: 'annual', "
            "'biennial', '2-year', '3-year', or an explicit duration. "
            "Free-text; emit verbatim when stated."
        ),
        examples=["annual", "biennial", "3-year"],
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Date of the next scheduled MDE review. Prefer ISO 8601 "
            "(YYYY-MM-DD); otherwise emit the date string verbatim as "
            "written in the source."
        ),
        examples=["2026-06-30", "June 2026"],
    )
    scan_type: Optional[str] = Field(
        default=None,
        description=(
            "How the radar's beam is mechanically or electronically steered. "
            "Typical values: "
            "CIRCULAR (continuous 360° mechanical rotation), "
            "SECTOR (back-and-forth sweep over a limited arc), "
            "RASTER (2D sweep covering an elevation stack), "
            "ELECTRONIC (phased-array beam steering, no moving parts), "
            "DWELL_AND_SWITCH (mechanical slew with pause at each beam "
            "position), "
            "HELICAL (continuous rotation with simultaneous elevation "
            "stepping). Emit as uppercase when possible."
        ),
        examples=["CIRCULAR", "ELECTRONIC", "DWELL_AND_SWITCH"],
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
