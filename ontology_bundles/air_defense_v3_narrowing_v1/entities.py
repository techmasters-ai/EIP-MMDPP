"""Canonical Pydantic entity classes for the air_defense_v3 bundle.

Plan v32 Phase 2 Tasks 14-19. Generated from ontology.yaml +
docs/design/2026-04-14-relationship-placement-table.md via
scripts/generate_entities_py.py.

Every class declares:
- ``model_config`` with ``ontology_name``, ``graph_id_fields``,
  ``identity_scope``, optional ``dodaf_parent``, and ``is_entity=True``.
- Required identity fields (no default).
- Optional non-identity fields with ``Field(description=..., examples=[...])``.
- Typed-edge fields via ``edge(label=..., ...)`` for entity-to-entity
  relationships per the placement table.

``ALL_ENTITIES`` registry at the bottom exposes {ontology_name: class}
for introspection consumers.

Intra-pass relationship DTO classes (RadarRelationship, MissileRelationship,
OtherSystemsRelationship) have been replaced by typed edges and are
removed. ``SystemLinkRelationship`` remains as the multi-pass DTO
exception per Decision 4 (lives in ``extraction_schemas/system_links.py``).
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def edge(
    label: str,
    *,
    description: str | None = None,
    examples: list | None = None,
    **field_kwargs: Any,
) -> Any:
    """Helper: declare a typed entity-to-entity edge field.

    Stores ``edge_label`` in ``json_schema_extra`` so introspection
    + contract tests can identify entity-to-entity fields. Otherwise
    delegates to ``Field(...)``.

    Args:
        label: The relationship type name (e.g. ``"HAS_ANTENNA"``).
            Must match a member of ``RelationshipType``.
        description: Forwarded to ``Field(description=...)``.
        examples: Forwarded to ``Field(examples=...)``.
        **field_kwargs: Forwarded to ``Field`` (``default``,
            ``default_factory``, etc.).
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


def profile_field(
    *,
    sections: list[str],
    subgroup: str,
    description: str,
    examples: list | None = None,
    default=None,
    default_factory=None,
    ge=None,
    le=None,
):
    """Field constructor for profile-mapped properties (spec §3.3 bucket 1).

    Tags the field with json_schema_extra={"profile_sections": [...],
    "profile_subgroup": "..."} so query_profiles' _project_field_groups
    can introspect and group by profile + subgroup.
    """
    extra = {"profile_sections": list(sections), "profile_subgroup": subgroup}
    kwargs: dict[str, Any] = {"description": description, "json_schema_extra": extra}
    if examples is not None:
        kwargs["examples"] = examples
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    else:
        kwargs["default"] = default
    if ge is not None:
        kwargs["ge"] = ge
    if le is not None:
        kwargs["le"] = le
    return Field(**kwargs)


def metadata_field(
    *,
    description: str,
    examples: list | None = None,
    default=None,
    sections: list[str] | None = None,
    subgroup: str | None = None,
):
    """Field constructor for system_metadata (spec §3.3 bucket 2).

    Real, indexed field — never surfaced by a starter profile. Used
    for audit trails, classifier IDs, status flags, review cadence.

    ``sections``/``subgroup`` are OPTIONAL, purely-additive projection
    tags. They let a metadata field ALSO surface in a query-profile
    section (e.g. ``governance``/``identification``) WITHOUT changing its
    bucket: the ``system_metadata`` marker is preserved, so the four-bucket
    contract still classifies the field as metadata. Extraction ignores
    both keys, so this is safe/additive.
    """
    extra: dict[str, Any] = {
        "profile_sections": list(sections or []),
        "system_metadata": True,
    }
    if subgroup is not None:
        extra["profile_subgroup"] = subgroup
    kwargs: dict[str, Any] = {
        "default": default,
        "description": description,
        "json_schema_extra": extra,
    }
    if examples is not None:
        kwargs["examples"] = examples
    return Field(**kwargs)


def identity_field(
    *,
    description: str,
    examples: list | None = None,
    default=None,
    sections: list[str] | None = None,
    subgroup: str | None = None,
):
    """Field constructor for identity adjuncts (spec §3.3 bucket 3).

    Used for fields like `nomenclature` and the missile-schema `name`
    that are identity context but not graph_id_fields. The four-bucket
    contract test treats them as identity rather than profile-mapped
    or metadata.

    ``sections``/``subgroup`` are OPTIONAL, purely-additive projection
    tags. They let an identity adjunct ALSO surface in a query-profile
    section (e.g. ``identification``) WITHOUT changing its bucket: the
    ``identity_field`` marker is preserved, so graph identity handling and
    the four-bucket contract are unaffected. Extraction ignores both keys.
    """
    extra: dict[str, Any] = {
        "profile_sections": list(sections or []),
        "identity_field": True,
    }
    if subgroup is not None:
        extra["profile_subgroup"] = subgroup
    kwargs: dict[str, Any] = {
        "default": default,
        "description": description,
        "json_schema_extra": extra,
    }
    if examples is not None:
        kwargs["examples"] = examples
    return Field(**kwargs)


# ----------------------------------------------------------------------
# Profile-section descriptions (SSoT)
# ----------------------------------------------------------------------
# One human-readable line per query-profile section. ``dossier`` is a
# catch-all that is NOT a literal field tag (it projects EVERY field that
# carries any profile_sections tag) — it is described here and advertised
# by the ontology service alongside the field-derived section names.
SECTION_DESCRIPTIONS: dict[str, str] = {
    "rf_parameters": "RF & waveform: frequency, PRI/PD, pulse coding, and antenna beam geometry.",
    "components": "Physical make-up: antennas, subsystems, stages, and hardware components.",
    "performance": "Operational metrics: power, scan, kinematics, and timing.",
    "engagement_envelope": "Engagement kinematics: intercept range, altitude ceiling/floor, speed, flight time, and guidance.",
    "governance": "Record lifecycle: status, responsible agency, and review schedule.",
    "identification": "Designators & reporting names: ELNOT, DIEQP, ASRD, and nomenclature.",
    "deployment": "Deployment context: the platform the system is mounted on or operated with.",
    "dossier": "Everything: all profiled fields across every section — the full entity dossier.",
}


# ----------------------------------------------------------------------
# Layer 1
# ----------------------------------------------------------------------
# confidence default convention: LLM-extracted entities default to None
# (the LLM may omit the field); anchor-walker-extracted entities (IMAGE,
# TEXT_BLOCK) hard-set 1.0 because extraction is deterministic.

class DocumentEntity(BaseModel):
    """Source Document — Technical manual, specification, report, drawing, or other source artifact
    """
    model_config = ConfigDict(ontology_name="DOCUMENT", graph_id_fields=["document_number"], identity_scope="global", dodaf_parent="DocumentResource", is_entity=True)

    document_number: str = Field(..., description="Official document designator (TM/MIL-STD/MIL-DTL number)", examples=['TM 9-1425-386-12', 'MIL-STD-1553B', 'MIL-DTL-31000G'])
    title: Optional[str] = Field(default=None, description="Full title of the document", examples=['Operator Manual for Patriot Missile System'])
    document_id: Optional[str] = Field(default=None, description="Internal system-assigned document identifier")
    classification: Optional[str] = Field(default=None, description="Security classification level", examples=['UNCLASSIFIED'])
    publication_date: Optional[str] = Field(default=None, description="Publication or revision date (YYYY-MM-DD)", examples=['2023-06-15'])
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the source file in the documents bucket")
    source_type: Optional[str] = Field(default=None, description="Category of the source document", json_schema_extra={"enum": ["MANUAL", "REPORT", "BRIEFING", "NOTE", "SCHEMATIC", "DRAWING", "SPECIFICATION", "CHECKLIST", "SPREADSHEET"]})
    issuing_org: Optional[str] = Field(default=None, description="Organization that published the document", examples=['U.S. Army TACOM'])
    language: Optional[str] = Field(default=None, description="Language of the document (ISO 639-1 code)", examples=['en'])
    workbook_name: Optional[str] = Field(default=None, description="Name of the spreadsheet workbook file", examples=['SA-20_MDE_Checklist.xlsx'])
    sheet_name: Optional[str] = Field(default=None, description="Name of the specific worksheet tab", examples=['Radar Parameters'])
    documents: List["DocumentEntity"] = edge(
        label="DERIVED_FROM",
        description="Source documents from which this document is derived or referenced.",
        examples=[["TM 9-1425-386-12", "MIL-STD-1553B"], ["MIL-DTL-31000G"]],
        default_factory=list,
    )
    documents: List["DocumentEntity"] = edge(
        label="SUPERSEDES",
        description="Older documents that this document supersedes or replaces.",
        examples=[["TM 9-1425-386-10"], ["MIL-STD-1553A"]],
        default_factory=list,
    )
    organizations: List["OrganizationEntity"] = edge(
        label="REVIEWED_BY",
        description="Organizations that reviewed or approved this document.",
        examples=[["Raytheon Missiles & Defense", "U.S. Army TACOM"], ["Department of Defense"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SectionEntity(BaseModel):
    """Document Section — A section or heading within a document
    """
    model_config = ConfigDict(ontology_name="SECTION", graph_id_fields=["section_number"], identity_scope="document", dodaf_parent="DocumentResource", is_entity=True)

    section_number: str = Field(..., description="Hierarchical section number within the document (docs:17235 R16-compliant identity)", examples=['3.2.1', '4.1', 'A-7'])
    heading: Optional[str] = Field(default=None, description="Section heading or title text", examples=['Chapter 3: Maintenance Procedures'])
    section_path: Optional[str] = Field(default=None, description="Full breadcrumb path to the section", examples=['Chapter 3 > Maintenance > Calibration'])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this section belongs to")
    page_start: Optional[int] = Field(default=None, description="Starting page number of the section", examples=[42])
    page_end: Optional[int] = Field(default=None, description="Ending page number of the section", examples=[67])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class FigureEntity(BaseModel):
    """Figure — A figure, diagram, or image within a document
    """
    model_config = ConfigDict(ontology_name="FIGURE", graph_id_fields=["figure_ref"], identity_scope="document", dodaf_parent="DocumentResource", is_entity=True)

    figure_ref: str = Field(..., description="Document-scoped figure reference (docs:17235 R16-compliant identity)", examples=['Figure 3-12', 'Fig. 4.1', 'Figure A-7'])
    figure_label: Optional[str] = Field(default=None, description="Human-friendly figure label or short title", examples=['Antenna Feed Assembly'])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this figure belongs to")
    caption: Optional[str] = Field(default=None, description="Figure caption text", examples=['Antenna Feed Assembly Exploded View'])
    page: Optional[int] = Field(default=None, description="Page number where the figure appears", examples=[55])
    figure_type: Optional[str] = Field(default=None, description="Category of the figure", json_schema_extra={"enum": ["BLOCK_DIAGRAM", "SCHEMATIC", "PHOTO", "SPECTRUM_PLOT", "WIRING_DIAGRAM", "FLOWCHART"]})
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the figure image. Always null in the initial change; populated once Artifact.self_ref plumbing lands.")
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class TableEntity(BaseModel):
    """Table — A data table within a document
    """
    model_config = ConfigDict(ontology_name="TABLE", graph_id_fields=["table_ref"], identity_scope="document", dodaf_parent="DocumentResource", is_entity=True)

    table_ref: str = Field(..., description="Document-scoped table reference (docs:17235 R16-compliant identity)", examples=['Table 4-1', 'Tbl. 2.3', 'Table A-2'])
    table_label: Optional[str] = Field(default=None, description="Human-friendly table label or short title", examples=['Frequency Parameters'])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this table belongs to")
    caption: Optional[str] = Field(default=None, description="Table caption or title", examples=['Radar System Frequency Parameters'])
    page: Optional[int] = Field(default=None, description="Page number where the table appears", examples=[78])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ImageEntity(BaseModel):
    """Image — An uncaptioned picture or embedded image within a document.

    Distinguished from FigureEntity by the absence of a "Figure N"-style
    caption: embedded logos, inline diagrams without labels, decorative
    photos. See docs/plans/2026-04-21-document-structure-pass-design.md §2.
    """
    model_config = ConfigDict(ontology_name="IMAGE", graph_id_fields=["image_ref"], identity_scope="document", dodaf_parent="DocumentResource", is_entity=True)

    image_ref: str = Field(..., description="Document-scoped self_ref of the picture (docs:17235 R16-compliant identity)", examples=["#/pictures/7", "#/pictures/12"])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this image belongs to")
    page: Optional[int] = Field(default=None, description="Page number where the image appears", examples=[1, 3])
    caption: Optional[str] = Field(default=None, description="Caption text when present, else None", examples=["Installation overview"])
    mime_type: Optional[str] = Field(default=None, description="MIME type of the backing image asset", examples=["image/png", "image/jpeg"])
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the picture bytes. Always null in the initial change; populated once Artifact.self_ref plumbing lands.")
    bbox: Optional[dict] = Field(default=None, description="Bounding box dict {l, t, r, b, page, coord_origin} from Docling provenance")
    image_role: Optional[str] = Field(default=None, description="Role heuristic derived from page position + caption", json_schema_extra={"enum": ["HEADER_LOGO", "INLINE_IMAGE", "UNCAPTIONED_FIGURE"]})
    confidence: Optional[float] = Field(default=1.0, description="Extraction confidence, 0–1. Anchor walker always emits 1.0.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


class TextBlockEntity(BaseModel):
    """TextBlock — A body-text paragraph emitted as neighbor context for FIGURE/IMAGE.

    Lazily emitted by the anchor walker: a TEXT_BLOCK only appears when at
    least one IMAGE or FIGURE declared it as a NEAR_TEXT neighbor. See
    docs/plans/2026-04-21-document-structure-pass-design.md §4.3.
    """
    model_config = ConfigDict(ontology_name="TEXT_BLOCK", graph_id_fields=["text_ref"], identity_scope="document", dodaf_parent="DocumentResource", is_entity=True)

    text_ref: str = Field(..., description="Document-scoped self_ref of the text item (docs:17235 R16-compliant identity)", examples=["#/texts/42", "#/texts/237"])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this text block belongs to")
    text: Optional[str] = Field(default=None, description="Rendered text content, truncated to 500 characters")
    label: Optional[str] = Field(default=None, description="Docling label (TEXT, PARAGRAPH, LIST_ITEM)", examples=["TEXT", "PARAGRAPH", "LIST_ITEM"])
    page: Optional[int] = Field(default=None, description="Page number where the text appears", examples=[3])
    confidence: Optional[float] = Field(default=1.0, description="Extraction confidence, 0–1. Anchor walker always emits 1.0.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

# ----------------------------------------------------------------------
# Layer 2
# ----------------------------------------------------------------------

class OrganizationEntity(BaseModel):
    """Organization — Contractor, program office, military branch, or government agency
    """
    model_config = ConfigDict(ontology_name="ORGANIZATION", graph_id_fields=["name"], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    name: str = Field(..., description="Full name of the organization", examples=['Raytheon Missiles & Defense', 'Lockheed Martin', 'U.S. Army TACOM'])
    org_type: Optional[str] = Field(default=None, description="Category of organization", json_schema_extra={"enum": ["PRIME_CONTRACTOR", "SUBCONTRACTOR", "PROGRAM_OFFICE", "MILITARY_BRANCH", "GOVERNMENT_AGENCY"]})
    cage_code: Optional[str] = Field(default=None, description="5-character CAGE code for the organization", examples=['58064'], pattern='^[A-Z0-9]{5}$')
    country: Optional[str] = Field(default=None, description="Country where the organization is headquartered", examples=['United States'])
    location: Optional[str] = Field(default=None, description="Primary facility location", examples=['Tucson, AZ'])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class PlatformEntity(BaseModel):
    """Platform — Vehicle, vessel, aircraft, or installation that hosts military systems
    """
    model_config = ConfigDict(ontology_name="PLATFORM", graph_id_fields=['name'], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    name: str = Field(..., description="Common name of the platform", examples=['SA-20 TEL', 'Patriot PAC-3 ICC'])
    platform_designation: Optional[str] = Field(default=None, description="Military designation of the platform", examples=['5P85SE'])
    platform_type: Optional[str] = Field(default=None, description="Category of the platform", json_schema_extra={"enum": ["AIRCRAFT", "SHIP", "GROUND_VEHICLE", "FIXED_SITE", "SPACE", "MOBILE_LAUNCHER", "AIR_DEFENSE_BATTERY"]})
    country: Optional[str] = Field(default=None, description="Country of origin or primary operator", examples=['Russia'])
    service_branch: Optional[str] = Field(default=None, description="Military branch operating the platform", examples=['Russian Aerospace Forces'])
    platform_status: Optional[str] = Field(default=None, description="Current lifecycle status of the platform", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    organizations: List["OrganizationEntity"] = edge(
        label="OPERATED_BY",
        description="Organizations that operate or crew this platform.",
        examples=[["U.S. Army", "NATO Integrated Air Defense Command"], ["Russian Aerospace Forces"]],
        default_factory=list,
    )
    organizations: List["OrganizationEntity"] = edge(
        label="MANUFACTURED_BY",
        description="Organizations that manufactured or produced this platform.",
        examples=[["Almaz-Antey"], ["Lockheed Martin", "Boeing"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="INSTANCE_OF",
        description="Platform archetypes or class definitions that this platform is an instance of.",
        examples=[["SA-20 TEL"], ["Patriot PAC-3 ICC"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class WeaponSystemEntity(BaseModel):
    """Weapon System — Generic weapon system (not radar, missile, or AAA specifically)
    """
    model_config = ConfigDict(ontology_name="WEAPON_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the weapon system", examples=['Phalanx CIWS', 'Bofors 40 mm L/70'])
    nomenclature: Optional[str] = Field(default=None, description="Military designation", examples=['Mk 15'])
    weapon_type: Optional[str] = Field(default=None, description="Category of weapon system", examples=['Close-In Weapon System'])
    subsystems: List["SubsystemEntity"] = edge(
        label="CONTAINS",
        description="Subsystems contained within this weapon system.",
        examples=[["Guidance Section", "Signal Processing Unit"], ["Antenna Array"]],
        default_factory=list,
    )
    subsystems: List["SubsystemEntity"] = edge(
        label="HAS_SUBSYSTEM",
        description="Major functional subsystems that make up this weapon system.",
        examples=[["Guidance Section", "Propulsion Assembly"], ["Fire Control Unit"]],
        default_factory=list,
    )
    components: List["ComponentEntity"] = edge(
        label="HAS_COMPONENT",
        description="Physical components that comprise this weapon system.",
        examples=[["PN-12345-A", "TWT-8090B"], ["CCA-0042"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="ENGAGES",
        description="Platforms or target types that this weapon system can engage.",
        examples=[["SA-20 TEL", "Patriot PAC-3 ICC"], ["MiG-29"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class EquipmentSystemEntity(BaseModel):
    """Equipment System — Top-level integrated weapon or defense system (generic)
    """
    model_config = ConfigDict(ontology_name="EQUIPMENT_SYSTEM", graph_id_fields=["name"], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    name: str = Field(..., description="Common name or designation of the system", examples=['Patriot PAC-3', 'THAAD', 'Aegis Combat System'])
    designation: Optional[str] = Field(default=None, description="Military AN/ designation or program designation", examples=['AN/MPQ-65'], pattern='^AN/[A-Z]{3}-\\d+')
    program_office: Optional[str] = Field(default=None, description="Managing program office", examples=['PEO Missiles and Space'])
    status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    prime_contractor: Optional[str] = Field(default=None, description="Lead contractor organization", examples=['Lockheed Martin'])
    service_branch: Optional[str] = Field(default=None, description="Military branch operating the system", examples=['U.S. Army'])
    components: List["ComponentEntity"] = edge(
        label="CONTAINS",
        description="Components physically contained within this equipment system.",
        examples=[["PN-12345-A", "TWT-8090B"], ["CCA-0042"]],
        default_factory=list,
    )
    subsystems: List["SubsystemEntity"] = edge(
        label="HAS_SUBSYSTEM",
        description="Major functional subsystems that comprise this equipment system.",
        examples=[["Guidance Section", "Signal Processing Unit"], ["Antenna Array"]],
        default_factory=list,
    )
    components: List["ComponentEntity"] = edge(
        label="HAS_COMPONENT",
        description="Physical components that are part of this equipment system.",
        examples=[["PN-12345-A"], ["TWT-8090B", "CCA-0042"]],
        default_factory=list,
    )
    organizations: List["OrganizationEntity"] = edge(
        label="MANUFACTURED_BY",
        description="Organizations that manufactured or produced this equipment system.",
        examples=[["Raytheon Missiles & Defense"], ["Lockheed Martin", "Northrop Grumman"]],
        default_factory=list,
    )
    equipment_systems: List["EquipmentSystemEntity"] = edge(
        label="INSTANCE_OF",
        description="Equipment system class or archetype that this system is an instance of.",
        examples=[["Patriot PAC-3"], ["THAAD"]],
        default_factory=list,
    )
    equipment_systems: List["EquipmentSystemEntity"] = edge(
        label="ALIAS_OF",
        description="Alternative names or designations referring to the same equipment system.",
        examples=[["Aegis Combat System"], ["AN/MPQ-65 Radar Set"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SubsystemEntity(BaseModel):
    """Subsystem — Major functional component within a military system
    """
    model_config = ConfigDict(ontology_name="SUBSYSTEM", graph_id_fields=["name"], identity_scope="document", dodaf_parent="MilitaryAsset", is_entity=True)

    name: str = Field(..., description="Name of the subsystem (document-scoped identity per spec §2.2)", examples=['Guidance Section', 'Signal Processing Unit', 'Antenna Array'])
    subsystem_role: Optional[str] = Field(default=None, description="Functional role within the parent system", examples=['Signal processing and target tracking'])
    part_number: Optional[str] = Field(default=None, description="Subsystem-level part or drawing number", examples=['GS-PAC3-001'])
    equipment_systems: List["EquipmentSystemEntity"] = edge(
        label="PART_OF",
        description="Equipment systems that this subsystem is part of.",
        examples=[["Patriot PAC-3", "THAAD"], ["Aegis Combat System"]],
        default_factory=list,
    )
    components: List["ComponentEntity"] = edge(
        label="HAS_COMPONENT",
        description="Physical components that are part of this subsystem.",
        examples=[["PN-12345-A", "TWT-8090B"], ["CCA-0042"]],
        default_factory=list,
    )
    radar_systems: List["RadarSystemEntity"] = edge(
        label="PART_OF",
        description="Radar systems that this subsystem is part of.",
        examples=[["Tombstone", "Clam Shell"], ["AN/MPQ-65"]],
        default_factory=list,
    )
    missile_systems: List["MissileSystemEntity"] = edge(
        label="PART_OF",
        description="Missile systems that this subsystem is part of.",
        examples=[["PAC-3 MSE", "SM-6 Block IA"], ["MIM-104F"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ComponentEntity(BaseModel):
    """Component — Individual physical or logical part within a subsystem or assembly
    """
    model_config = ConfigDict(ontology_name="COMPONENT", graph_id_fields=["part_number"], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    part_number: str = Field(..., description="Manufacturer-assigned part number (docs:17235 R16-compliant identity)", examples=['PN-12345-A', 'TWT-8090B', 'CCA-0042'], pattern='^[A-Z0-9][A-Z0-9\\-/]{2,20}$')
    name: Optional[str] = Field(default=None, description="Common name of the component", examples=['Traveling Wave Tube'])
    component_type: Optional[str] = Field(default=None, description="Category or class of the component", examples=['Amplifier'])
    nsn: Optional[str] = Field(default=None, description="National Stock Number in NNNN-NN-NNN-NNNN format", examples=['5961-01-234-5678'], pattern='^\\d{4}-\\d{2}-\\d{3}-\\d{4}$')
    cage_code: Optional[str] = Field(default=None, description="5-character Commercial and Government Entity code identifying the manufacturer", examples=['1ABC3'], pattern='^[A-Z0-9]{5}$')
    manufacturer: Optional[str] = Field(default=None, description="Name of the component manufacturer", examples=['L3Harris Technologies'])
    material: Optional[str] = Field(default=None, description="Primary material composition of the component", examples=['Aluminum 7075-T6'])
    weight_kg: Optional[float] = Field(default=None, description="Weight of the component in kilograms", examples=[2.5])
    subsystems: List["SubsystemEntity"] = edge(
        label="PART_OF",
        description="Subsystems or assemblies that this component is part of.",
        examples=[["Guidance Section", "Signal Processing Unit"], ["Antenna Array"]],
        default_factory=list,
    )
    organizations: List["OrganizationEntity"] = edge(
        label="MANUFACTURED_BY",
        description="Organizations that manufactured or produced this component.",
        examples=[["L3Harris Technologies"], ["Raytheon Missiles & Defense", "Northrop Grumman"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RadarSystemEntity(BaseModel):
    """Radar System — Active or passive radar system (search, track, fire control, SAR, AESA, PESA)
    """
    model_config = ConfigDict(ontology_name="RADAR_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Canonical designation of the RADAR itself. Accept canonical proper-noun radar names from prose when unambiguous (e.g. 'Fan Song', 'Spoon Rest', 'Tombstone', 'Flap Lid', 'AN/MPQ-65'). FORBIDDEN values — never emit any of these as system_name because they are weapon/missile systems, not radars: SA-2, SA-3, SA-5, SA-6, SA-10, SA-12, SA-15, SA-17, SA-20, SA-21, SA-22, SA-23, Patriot, PAC-2, PAC-3, PAC-3 MSE, Hawk, Nike-Hercules, S-75, S-125, S-200, S-300, S-350, S-400, S-500, Aegis BMD, SM-2, SM-3, SM-6, THAAD, Arrow, Iron Dome, David's Sling. Also FORBIDDEN: aircraft / platform / target names (U-2, SR-71, RF-4C, F-4, F-15, F-16, B-52, MiG-21, MiG-23, MiG-29, Su-27) — these are targets that radars detect, not radars themselves. If the text says 'the SA-2 radar', emit the radar's own name ('Fan Song') if stated, otherwise omit. Do NOT emit 'SA-2' here. Reject descriptive phrases ('the radar', 'the acquisition radar').", examples=['Fan Song', 'Spoon Rest', 'Tombstone', 'AN/MPQ-65', 'Flap Lid'])

    # ===== Flat-checklist fields (spec §3.3) =====
    # waveform group — sections=['rf_parameters']
    nominal_rf_mhz: Optional[float] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Nominal operating RF (carrier frequency) in megahertz. Common radar bands: L-band 1000-2000, S-band 2000-4000, C-band 4000-8000, X-band 8000-12000, Ku-band 12000-18000. If the source gives GHz, multiply by 1000. If it gives a range, emit the center frequency.',
        examples=[3000, 9400, 16000],
    )
    frequency_excursion_mhz: Optional[float] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Frequency excursion (chirp bandwidth) in megahertz — for an LFM chirp waveform, the total swept bandwidth Δf across the pulse duration. Determines range resolution: resolution_m ≈ 150 / bandwidth_MHz. Only meaningful for frequency-modulated (chirped) waveforms.',
        examples=[1.0, 10.0, 50.0],
    )
    nominal_pri_usec: Optional[float] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Nominal Pulse Repetition Interval (PRI) in microseconds — the time from the start of one pulse to the start of the next. PRI = 1 / PRF. Determines unambiguous range: unambiguous_range_km = PRI_usec × 0.15. Typical long-range search radars: 3000-10000 µs. Fire-control radars: 100-1000 µs. If the source gives PRF in Hz, convert: PRI_usec = 1_000_000 / PRF_Hz.',
        examples=[1000, 5000],
    )
    nominal_pd_usec: Optional[float] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Nominal Pulse Duration / pulse width (PD) in microseconds — the length of a single transmitted pulse. Short pulses (0.1-1 µs) give fine range resolution; long pulses (50-200 µs) give more energy on target. Compressed-pulse radars specify the pre-compression PD here (the long pulse before matched filtering).',
        examples=[0.5, 50.0, 200.0],
    )
    inter_pulse: Optional[str] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Inter-pulse modulation — how successive pulses vary. Typical patterns: CONSTANT_PRI (fixed spacing between pulses), PRI_STAGGER (multi-level PRI that cycles between values; resolves range ambiguity), PRI_JITTER (random PRI variation; anti-jam / ECCM), FREQ_AGILE (pulse-to-pulse frequency hopping). Emit as uppercase token when possible.',
        examples=['CONSTANT_PRI', 'PRI_STAGGER', 'FREQ_AGILE'],
    )
    pulses_per_dwell: Optional[int] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Number of pulses coherently or non-coherently integrated in one beam-position dwell. More pulses = more energy on target (and better Doppler resolution for coherent integration), at the cost of slower scan rate. Typical values: 8-64 for modern radars.',
        examples=[16, 64],
    )
    dwell_time: Optional[str] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description="Time spent at a single beam position (dwell-and-switch scans). Free-text because sources sometimes give a duration ('12 ms'), a count ('16 pulses'), or a descriptive phrase. Only relevant for DWELL_AND_SWITCH or phased-array scan_types.",
        examples=['12 ms', '16 pulses'],
    )
    intra_pulse_mop: Optional[str] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Intra-pulse modulation (Modulation On Pulse) — how the transmitted pulse is modulated for compression. Typical values: CW (continuous wave / unmodulated), LFM_CHIRP (linear frequency modulation / chirp), NLFM (non-linear FM), BARKER_CODE (binary phase-coded, Barker sequence), POLYPHASE (multi-level phase code, e.g. Frank / P1-P4), BIPHASE (2-level phase code). Free-text emission OK if the source uses different terms.',
        examples=['LFM_CHIRP', 'BARKER_CODE', 'BIPHASE'],
    )
    num_bits_in_code: Optional[int] = profile_field(
        sections=['rf_parameters'], subgroup='waveform',
        description='Number of chips in the phase-code sequence used for pulse compression. Only meaningful for phase-coded waveforms (Barker, polyphase, etc.). Common Barker codes: 7, 11, 13. Longer codes (128, 256, 1024) appear in modern systems. Null for CW or FM-chirp radars.',
        examples=[7, 13, 1024],
    )
    # antenna group — sections=['rf_parameters', 'components']
    antenna_dim_az_m: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description="Antenna aperture width in the azimuth (horizontal) dimension, in meters. For a rectangular planar array this is the long dimension of the face; for a parabolic dish it's the horizontal diameter. Drives azimuth beamwidth via beamwidth ≈ (wavelength / aperture) × 51 degrees.",
        examples=[4.5, 12.0],
    )
    antenna_dim_el_m: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description='Antenna aperture height in the elevation (vertical) dimension, in meters. Analogous to antenna_dim_az_m but for the elevation plane. A radar with asymmetric az/el dimensions has a fan beam (narrow azimuth, broad elevation) typical of 2D surveillance radars.',
        examples=[2.5, 4.0],
    )
    beamwidth_az_deg: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description="Main-beam 3dB azimuth beamwidth, in degrees. Defines the angular spread of the radar's beam in the horizontal plane between half-power points. Typical ground-based S-band / C-band radars: 1-4 degrees.",
        examples=[1.5, 2.8],
    )
    beamwidth_el_deg: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description='Main-beam 3dB elevation beamwidth, in degrees. The vertical analogue of beamwidth_az_deg. Fan-beam surveillance radars have intentionally wide elevation beamwidths (10-40°) to cover airspace with a single scan; pencil-beam tracking radars keep elevation beamwidth tight (1-4°).',
        examples=[1.5, 15.0],
    )
    gain_dbi: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description='Peak antenna gain in dBi (decibels relative to an isotropic radiator). Higher gain = narrower main beam. Typical dish / parabolic radar antennas: 30-45 dBi. Phased arrays: 35-50 dBi.',
        examples=[38.0, 42.0],
    )
    antenna_photo: Optional[bool] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description='Whether an antenna photograph is included in the record (Y/N). Use null when the document does not state this — do NOT default to false. Emit true only when the text explicitly indicates a photograph is included.',
    )
    spoiled: Optional[bool] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description='Whether the beam is spoiled (Y/N). Use null when the document does not state this — do NOT default to false. Emit true only when the text explicitly indicates a spoiled beam; emit false only when the text explicitly indicates an unspoiled beam.',
    )
    coverage_limits_el_deg: Optional[float] = profile_field(
        sections=['rf_parameters', 'components'], subgroup='antenna',
        description="Maximum elevation angle (degrees) the radar can scan or track to. Ground-based search radars typically cap at 30-40°; fire-control radars can go to 80°+. When the source gives a range ('0-45°'), emit the upper limit. Null when the document doesn't state a cap.",
        examples=[45.0, 80.0],
    )
    # transmit group — sections=['rf_parameters', 'performance']
    tx_peak_power_kw: Optional[float] = profile_field(
        sections=['rf_parameters', 'performance'], subgroup='transmit',
        description="Transmitter peak power output in kilowatts. This is the power at the transmitter's output port BEFORE the antenna — not effective radiated power. Typical S-band ground radars: 100-1000 kW peak. If the source gives power in watts, divide by 1000; if in megawatts, multiply by 1000.",
        examples=[150, 600, 1000],
    )
    erp_dbw: Optional[float] = profile_field(
        sections=['rf_parameters', 'performance'], subgroup='transmit',
        description='Effective Radiated Power in dBW (decibels relative to 1 watt). ERP = transmitter power × antenna gain, expressed on the log scale. Typical combat radars: 50-90 dBW. If the source gives ERP in dBm, subtract 30 (1 W = 30 dBm = 0 dBW). If given in watts or kilowatts, convert: dBW = 10 × log10(watts).',
        examples=[72.0, 85.0],
    )
    # scan group — sections=['rf_parameters', 'performance']
    scan_type: Optional[str] = profile_field(
        sections=['rf_parameters', 'performance'], subgroup='scan',
        description="How the radar's beam is mechanically or electronically steered. Typical values: CIRCULAR (continuous 360° mechanical rotation), SECTOR (back-and-forth sweep over a limited arc), RASTER (2D sweep covering an elevation stack), ELECTRONIC (phased-array beam steering, no moving parts), DWELL_AND_SWITCH (mechanical slew with pause at each beam position), HELICAL (continuous rotation with simultaneous elevation stepping). Emit as uppercase when possible.",
        examples=['CIRCULAR', 'ELECTRONIC', 'DWELL_AND_SWITCH'],
    )
    scan_period_sec: Optional[float] = profile_field(
        sections=['rf_parameters', 'performance'], subgroup='scan',
        description='Time (seconds) to complete one full scan pattern — e.g. 360° rotation for a CIRCULAR scan, or one full raster for a RASTER scan. Combined with beamwidth this determines revisit rate. Typical rotating search radars: 4-12 s per revolution.',
        examples=[4.0, 10.0],
    )
    # classification group — sections=['rf_parameters']
    emitter_function: Optional[str] = profile_field(
        sections=['rf_parameters'], subgroup='classification',
        description="Operational role of the radar in an engagement kill-chain. Enum values and their meanings: SEARCH = early-warning / acquisition radar that detects targets at long range; TRACKING = radar that maintains target track after acquisition but does not provide the terminal weapon-guidance function; FIRE_CONTROL = terminal-guidance radar that provides the tracking signal used by the weapon system's seeker or command-guidance link. Guidance / illumination radars such as Fan Song belong here; MULTI_FUNCTION = a single radar that performs multiple roles (phased-array designs like AN/SPY-1 are typical examples); HEIGHT_FINDER = dedicated elevation-measurement radar paired with 2D search radars; NAV = navigation or weather radar (not a combat emitter).",
        examples=['SEARCH', 'FIRE_CONTROL', 'TRACKING', 'MULTI_FUNCTION'],
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        sections=['identification'], subgroup='designators',
        description="Official military nomenclature — the formal alphanumeric designation assigned by the manufacturing country. For US radars this is the JETDS / AN-style designator (e.g. 'AN/MPQ-65'). For Russian / Soviet-origin radars it's the GRAU index or manufacturer model (e.g. '5N63S', '30N6E'). Distinct from system_name, which is the common (often NATO reporting) name. Emit when the document explicitly states the formal designation alongside the common name.",
        examples=['AN/MPQ-65', '5N63S', '30N6E', 'AN/SPY-1D'],
    )

    # System metadata
    elnot: Optional[str] = metadata_field(
        sections=['identification'], subgroup='designators',
        description='ELINT Notation (ELNOT) — an ELINT-community unique alphabetic code assigned to a specific emitter signal by signals intelligence databases (typically a 4- or 5-letter code). Only appears in intelligence-community source documents. Emit verbatim from the document — do not infer.',
    )
    dieqp: Optional[str] = metadata_field(
        sections=['identification'], subgroup='designators',
        description='Digital Intelligence Equipment Parameters (DIEQP) identifier — a cross-reference ID into the DIEQP database maintained by the MDE (Mission Data Engineering) community. Typically a short alphanumeric token. Only appears in IC / MDE source documents. Emit verbatim — do not infer.',
    )
    asrd: Optional[str] = metadata_field(
        sections=['identification'], subgroup='designators',
        description='ASRD identifier — a catalog code from the All-Source Reference Document, a classified IC catalog of emitters. Emit verbatim when explicitly stated in the source; do not infer or cross-reference.',
    )
    system_status: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description='Lifecycle status of the radar system as described in the source. Typical values: OPERATIONAL (currently deployed), DEVELOPMENTAL (prototype or pre-IOC), RETIRED (withdrawn from service), UPGRADED (modified variant superseding the base model), EXPORTED (sold to foreign operators only). Emit only when the document explicitly states the status; do not infer OPERATIONAL from historical narrative or from the fact that the radar appears in a museum display.',
        examples=['OPERATIONAL', 'RETIRED', 'DEVELOPMENTAL'],
    )
    responsible_agency: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description="Organization responsible for maintaining the MDE record for this radar. Typically a 3-letter IC acronym (e.g. 'IWC' = Information Warfare Center, 'NASIC' = National Air and Space Intelligence Center, 'ONI' = Office of Naval Intelligence, 'NGIC' = National Ground Intelligence Center).",
        examples=['IWC', 'NASIC', 'ONI', 'NGIC'],
    )
    review_cycle: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description="Scheduled cadence at which the MDE record for this radar is reviewed and re-validated. Typical values: 'annual', 'biennial', '2-year', '3-year', or an explicit duration. Free-text; emit verbatim when stated.",
        examples=['annual', 'biennial', '3-year'],
    )
    next_review_date: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description='Date of the next scheduled MDE review. Prefer ISO 8601 (YYYY-MM-DD); otherwise emit the date string verbatim as written in the source.',
        examples=['2026-06-30', 'June 2026'],
    )

    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this radar system is installed.",
        examples=["SA-20 TEL", "Patriot PAC-3 ICC"],
        default=None,
        json_schema_extra={"profile_sections": ["deployment"], "profile_subgroup": "deployment"},
    )
    equipment_systems: List["EquipmentSystemEntity"] = edge(
        label="IS_A",
        description="Equipment system categories or types that this radar system is a member of.",
        examples=[["Patriot PAC-3", "THAAD"], ["Aegis Combat System"]],
        default_factory=list,
    )
    radar_systems: List["RadarSystemEntity"] = edge(
        label="ALIAS_OF",
        description="Alternative names or designations for this radar system.",
        examples=[["AN/MPQ-65 Radar Set"], ["Tombstone"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="TRACKS",
        description="Platforms or target types that this radar system tracks.",
        examples=[["SA-20 TEL"], ["MiG-29", "Ballistic Missile"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="DETECTS",
        description="Platforms or target types that this radar system detects.",
        examples=[["MiG-29", "Cruise Missile"], ["Ballistic Missile"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="DESIGNATES",
        description="Platforms or targets that this radar system designates for engagement.",
        examples=[["SA-20 TEL", "MiG-29"], ["Ballistic Missile"]],
        default_factory=list,
    )
    missile_systems: List["MissileSystemEntity"] = edge(
        label="SUPPORTS_ENGAGEMENT_OF",
        description="Missile systems whose engagement this radar system supports.",
        examples=[["PAC-3 MSE", "SM-6 Block IA"], ["THAAD Interceptor"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Overall extraction confidence for this radar instance, 0-1. Combines identity certainty + parametric confidence. Use 0.9-1.0 when identity is from a table/figure caption and parameters are explicit; 0.5-0.8 for prose mentions with partial parameters; <0.5 for inferred / reconstructed values. System-populated — leave null if unsure.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class MissileSystemEntity(BaseModel):
    """Missile System — Guided missile weapon system with seeker, guidance, and propulsion
    """
    model_config = ConfigDict(ontology_name="MISSILE_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Canonical designation of the MISSILE / WEAPON SYSTEM itself. Accept canonical proper-noun weapon identifiers from prose when unambiguous (e.g. 'SA-2', 'SA-20', 'PAC-3 MSE', '9M96'). FORBIDDEN values — never emit any of these as system_name because they are radars, not missile/weapon systems: Fan Song, Spoon Rest, Flat Face, Side Net, Flap Lid, Grave Stone, Big Bird, Back Trap, Tombstone, AN/MPQ-53, AN/MPQ-65, AN/SPY-1, AN/SPY-6, AN/TPY-2. Also FORBIDDEN: aircraft / platform / target names (U-2, SR-71, RF-4C, F-4, F-15, F-16, B-52, MiG-21, MiG-23, MiG-29, Su-27) — these are aircraft that may be engaged by a missile, but they are NOT missile systems. If the text says 'U-2 was shot down by SA-2', only emit the missile ('SA-2'). Do NOT emit 'U-2'. Reject descriptive phrases ('the missile', 'the interceptor').", examples=['SA-2', 'SA-20', 'PAC-3 MSE', 'SM-6 Block IA', '9M96'])

    # ===== Flat-checklist fields (spec §3.3) =====
    # airframe group — sections=['components', 'performance']
    body_length_m: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='airframe',
        description='Overall missile body length in meters, nose-tip to tail (booster included if permanently attached). Typical SAM ranges vary by class. If source gives feet, multiply by 0.3048.',
    )
    body_diameter_m: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='airframe',
        description='Missile body diameter (airframe cross-section) in meters. If source gives inches, multiply by 0.0254; if centimeters, divide by 100.',
    )
    total_mass_kg: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='airframe',
        description="Total missile launch mass in kilograms (all stages + warhead + fuel). If source gives pounds, divide by 2.205. Note: the MDE checklist unit column shows 'deg' in row 20 — that's a source typo; this field is mass in kg.",
    )
    missile_photo: Optional[bool] = profile_field(
        sections=['components', 'performance'], subgroup='airframe',
        description='Whether a missile photograph is included in the record (Y/N). Use null when the document does not state this — do NOT default to false. Emit true only when the text explicitly indicates a photograph is included.',
    )
    # seeker group — sections=['components', 'performance']
    seeker_type: Optional[str] = profile_field(
        sections=['components', 'performance'], subgroup='seeker',
        description="Terminal-phase seeker technology. Enum values: ACTIVE_RADAR = onboard radar illuminator + receiver (PAC-3, AMRAAM); SEMI_ACTIVE_RADAR = receives target echoes from a separate illuminating radar; PASSIVE_RADAR = detects target's own RF emissions (anti-radiation missiles); IR = infrared / thermal guidance; DUAL_MODE = IR + radar; ARM = anti-radiation homing; GPS_INS = navigation-based (no terminal seeker); COMMAND = no onboard seeker, ground-uplink guidance only from an external control unit.",
    )
    # booster group — sections=['components', 'performance']
    booster_time_sec: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='booster',
        description='Booster (first-stage) motor burn duration in seconds. The booster provides initial acceleration from rest to a target velocity at which the sustainer can take over. Typical SAM boosters: 4-10 s.',
        examples=[6.0, 8.0],
    )
    booster_thrust: Optional[str] = profile_field(
        sections=['components', 'performance'], subgroup='booster',
        description='Booster-stage thrust. Free-text (see ejector_thrust). Emit verbatim with units as the source provides.',
        examples=['220 kN', '50000 lbf'],
    )
    booster_mass_kg: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='booster',
        description='Booster stage mass (hardware + propellant) in kilograms. For two-stage missiles the booster typically separates after burnout. Typical: 300-1000 kg for medium / long-range SAMs.',
        examples=[700.0, 1200.0],
    )
    # sustain group — sections=['components', 'performance']
    sustain_time_sec: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='sustain',
        description='Sustainer (second-stage) motor burn duration in seconds. Sustains the velocity gained from the booster and maintains kinetic energy through the intercept geometry. Typical: 10-30 s for long-range SAMs.',
        examples=[18.0, 25.0],
    )
    sustain_thrust: Optional[str] = profile_field(
        sections=['components', 'performance'], subgroup='sustain',
        description='Sustainer-stage thrust. Free-text (see ejector_thrust). Typically lower than booster thrust because the sustainer burns longer at lower pressure.',
        examples=['50 kN', '11000 lbf'],
    )
    sustain_mass_kg: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='sustain',
        description='Sustainer stage mass (hardware + propellant) in kilograms. For single-stage missiles this is the whole motor. Typical range: 100-500 kg for medium / long-range SAMs.',
        examples=[250.0, 400.0],
    )
    # ejector group — sections=['components', 'performance']
    ejector_time_sec: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='ejector',
        description='Ejector / launch-eject stage duration in seconds. The ejector is a low-thrust charge that pushes the missile out of a vertical launch canister before main motor ignition (cold launch). Typical: 0.5-2 s. Null if the missile uses hot launch (no ejector stage).',
        examples=[1.0, 1.5],
    )
    ejector_thrust: Optional[str] = profile_field(
        sections=['components', 'performance'], subgroup='ejector',
        description='Ejector-stage thrust. Free-text because the checklist unit column is blank — source may give newtons, kilonewtons, or pounds-force. Emit verbatim with the unit the source uses.',
        examples=['12 kN', '2700 lbf'],
    )
    ejector_mass_kg: Optional[float] = profile_field(
        sections=['components', 'performance'], subgroup='ejector',
        description='Mass of the ejector stage + its expended propellant, in kilograms. Separates from the missile and is discarded before main motor burn. Typical: 5-30 kg.',
        examples=[20.0, 10.0],
    )
    # engagement group — sections=['performance']
    min_intercept_km: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='engagement',
        description="Minimum effective intercept / engagement range, in kilometers. Below this range the missile's safety arm, initialization sequence, or terminal-guidance lock-on cannot complete in time. Typical SAMs: 2-10 km minimum. If source gives the value in miles, nautical miles, feet, or meters, convert to km before emitting. Do not copy the raw source number when the source unit is not already kilometers.",
        examples=[2.0, 5.0],
    )
    max_intercept_km: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='engagement',
        description="Maximum effective intercept / engagement range, in kilometers. The outer edge of the missile's engagement envelope against an assumed target profile. Use the document's effective range value here. Do NOT use slant range, ferry range, or maximum kinematic distance unless the document explicitly says those are the effective engagement range. Convert source units to km.",
        examples=[35.0, 400.0],
    )
    min_altitude_km: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='engagement',
        description='Minimum engagement altitude, in kilometers. Below this altitude the missile cannot acquire or intercept. Legacy SAMs (SA-2) have ~1 km minimum; modern systems reach sea-skimming altitudes (<0.05 km). Only populate when the document explicitly states a minimum altitude / floor; do not infer from generic system knowledge.',
        examples=[0.05, 1.0],
    )
    max_altitude_km: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='engagement',
        description="Maximum engagement altitude (ceiling), in kilometers. The top of the missile's engagement envelope. Classical high-altitude SAMs (SA-2 / S-75): ~25 km. Exo-atmospheric interceptors (SM-3, GBI): >100 km. If the source gives ceiling in feet or meters, convert to km.",
        examples=[18.0, 35.0, 180.0],
    )
    max_launch_angle_deg: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='engagement',
        description="Maximum launch angle off vertical / elevation, in degrees. For vertical-launch systems this may be fixed at 0° (true vertical) or slightly canted. For tilt-launchers (SA-2, SA-3) it's typically 50-80°. 90° means the launcher can fire horizontally.",
        examples=[60.0, 80.0],
    )
    # kinematics group — sections=['performance']
    average_speed_mps: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Average in-flight speed in meters per second (averaged over powered + coast phases). If source gives Mach, multiply by ~340 (sea-level Mach ≈ 340 m/s). If km/h, divide by 3.6.',
    )
    max_speed_mps: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Peak in-flight speed in meters per second (typically reached at end of boost phase). Same unit-conversion rules as average_speed_mps.',
    )
    max_flyout_time_sec: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Maximum total time of flight from launch to intercept or self-destruct, in seconds. Determined by fuel burn + coast dynamics + range. Typical long-range SAMs: 60-120 s.',
        examples=[60.0, 120.0],
    )
    flight_time_sec: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Typical / nominal flight time from launch to expected intercept, in seconds (for a median engagement profile). Shorter than max_flyout_time_sec, which is the worst-case.',
        examples=[30.0, 60.0],
    )
    coast_time_sec: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Duration of the unpowered (post-motor-burnout) coast phase, in seconds. The missile relies on residual kinetic energy + aerodynamic control. Longer-range missiles have more coast time; short-range MANPADS may have near-zero coast.',
        examples=[10.0, 45.0],
    )
    total_burn_time_sec: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description='Total powered-flight burn time across all propulsion stages (boost + sustain, plus ejector if applicable), in seconds. Excludes any coast phase. Typical long-range two-stage SAMs: 20-40 s total.',
        examples=[22.0, 35.0],
    )
    intra_salvo_time_sec: Optional[float] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='kinematics',
        description="Time between successive missile launches in a salvo, in seconds. For multi-missile engagements (shoot-look-shoot or ripple-fire tactics). Note: the MDE checklist labels this 'Intra-Solvo Time' — source typo.",
        examples=[6.0, 30.0],
    )
    # guidance group — sections=['performance']
    guidance_type: Optional[str] = profile_field(
        sections=['performance', 'engagement_envelope'], subgroup='guidance',
        description='Primary guidance method used to drive the missile to intercept. Enum values and meanings: COMMAND = ground station computes aim and uplinks guidance from an external control unit; BEAM_RIDING = missile rides the radar beam to target; SARH = semi-active radar homing (missile homes on target-illuminated RF energy; target illumination from a separate radar); ARH = active radar homing (missile carries its own seeker radar); IR = passive infrared homing; TVM = track-via-missile (missile relays target data back to ground, hybrid command + SARH); GPS_INS = inertial-navigation with GPS updates (usually for mid-course of a longer-range missile); DUAL_MODE = combines two modes (e.g. IR + radar). Many modern missiles combine phases: MID_COURSE + TERMINAL.',
    )
    # classification group — sections=['performance']
    emitter_function: Optional[str] = profile_field(
        sections=['performance'], subgroup='classification',
        description="MDE-checklist emitter-function field. For a missile weapon system this is typically null (missiles don't usually have their own active emitters; their radar is a separate system). Emit a value only when the source explicitly assigns an emitter function to the missile itself (e.g. missile seeker listed as 'ACTIVE_RADAR_HOMING').",
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        sections=['identification'], subgroup='designators',
        description='Military designation or NATO reporting name.',
        examples=['MIM-104F'],
    )
    name: Optional[str] = identity_field(
        sections=['identification'], subgroup='designators',
        description="Formal NAME field from the MDE checklist, distinct from the common ``system_name``. Often the full proper name (e.g. 'Patriot Advanced Capability 3 Missile Segment Enhancement'). Emit when the source provides a formal long-form name alongside the short system_name.",
        examples=['Patriot Advanced Capability 3 MSE', 'S-400 Triumf'],
    )

    # System metadata
    dieqp: Optional[str] = metadata_field(
        sections=['identification'], subgroup='designators',
        description='Digital Intelligence Equipment Parameters (DIEQP) identifier — a cross-reference ID into the DIEQP database maintained by the MDE (Mission Data Engineering) community. Only appears in IC / MDE source documents. Emit verbatim — do not infer.',
    )
    asrd: Optional[str] = metadata_field(
        sections=['identification'], subgroup='designators',
        description='ASRD identifier — a catalog code from the All-Source Reference Document, a classified IC catalog. Emit verbatim when explicitly stated; do not infer.',
    )
    system_status: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description='Lifecycle status of the missile system. Typical values: OPERATIONAL (currently deployed), DEVELOPMENTAL (prototype or pre-IOC), RETIRED (withdrawn from service), UPGRADED (modified variant superseding a base model), EXPORTED (sold to foreign operators only). Emit only when the document explicitly states the status; do not infer it from historical or descriptive text.',
        examples=['OPERATIONAL', 'RETIRED', 'DEVELOPMENTAL'],
    )
    responsible_agency: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description="Organization responsible for maintaining the MDE record for this missile system. Typically a 3-letter IC acronym (e.g. 'IWC' = Information Warfare Center, 'NASIC' = National Air and Space Intelligence Center, 'ONI' = Office of Naval Intelligence, 'NGIC' = National Ground Intelligence Center, 'MSIC' = Missile and Space Intelligence Center).",
        examples=['IWC', 'NASIC', 'MSIC'],
    )
    review_cycle: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description="Scheduled cadence at which the MDE record for this missile is reviewed. Typical values: 'annual', 'biennial', '2-year', '3-year', or an explicit duration. Free-text; emit verbatim.",
        examples=['annual', 'biennial', '3-year'],
    )
    next_review_date: Optional[str] = metadata_field(
        sections=['governance'], subgroup='lifecycle',
        description='Date of the next scheduled MDE review. Prefer ISO 8601 (YYYY-MM-DD); otherwise emit the date string verbatim.',
        examples=['2026-06-30', 'June 2026'],
    )

    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this missile system is installed.",
        examples=["M903 Launching Station", "SA-20 TEL"],
        default=None,
        json_schema_extra={"profile_sections": ["deployment"], "profile_subgroup": "deployment"},
    )
    platforms: List["PlatformEntity"] = edge(
        label="DEFENDS",
        description="Platforms or areas that this missile system defends.",
        examples=[["Patriot PAC-3 ICC", "Mobile Radar Site"], ["Forward Operating Base"]],
        default_factory=list,
    )
    weapon_systems: List["WeaponSystemEntity"] = edge(
        label="IS_A",
        description="Weapon system categories that this missile system is a member of.",
        examples=[["Phalanx CIWS"], ["Bofors 40 mm L/70"]],
        default_factory=list,
    )
    missile_systems: List["MissileSystemEntity"] = edge(
        label="ALIAS_OF",
        description="Alternative names or designations for this missile system.",
        examples=[["PAC-3 MSE"], ["MIM-104F"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="ENGAGES",
        description="Platforms or target types that this missile system can engage.",
        examples=[["MiG-29", "Cruise Missile"], ["Ballistic Missile"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Overall extraction confidence for this missile instance, 0-1. System-populated field. Leave null unless the document itself explicitly provides a confidence value.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class AirDefenseArtillerySystemEntity(BaseModel):
    """Air Defense Artillery System — Gun-based air defense system (AAA)
    """
    model_config = ConfigDict(ontology_name="AIR_DEFENSE_ARTILLERY_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the AAA system", examples=['ZSU-23-4 Shilka', 'Gepard FlakPanzer'])
    DIEQP: Optional[str] = Field(default=None, description="Defense Intelligence Equipment identifier code", examples=["DE12345", "DE67890"])
    caliber: Optional[str] = Field(default=None, description="Gun barrel caliber", examples=['23 mm'])
    max_tactical_range: Optional[str] = Field(default=None, description="Maximum tactical engagement range", examples=['2.5 km'])
    max_vertical_range: Optional[str] = Field(default=None, description="Maximum vertical engagement range (ceiling)", examples=['1.5 km'])
    min_vertical_range: Optional[str] = Field(default=None, description="Minimum vertical engagement range", examples=['100 m'])
    max_horizontal_range: Optional[str] = Field(default=None, description="Maximum horizontal range", examples=['2.5 km'])
    min_horizontal_range: Optional[str] = Field(default=None, description="Minimum horizontal range", examples=['200 m'])
    acquisition_delay: Optional[str] = Field(default=None, description="Time from cue to target acquisition", examples=['3 s'])
    handoff_delay: Optional[str] = Field(default=None, description="Time for target handoff between systems", examples=['2 s'])
    track_delay: Optional[str] = Field(default=None, description="Time from acquisition to stable track", examples=['1.5 s'])
    muzzle_velocity: Optional[str] = Field(default=None, description="Projectile muzzle velocity", examples=['970 m/s'])
    maximum_rate_of_fire: Optional[str] = Field(default=None, description="Maximum cyclic rate of fire", examples=['3400 rounds/min'])
    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this AAA system is installed.",
        examples=["SA-20 TEL", "Patriot PAC-3 ICC"],
        default=None,
    )
    weapon_systems: List["WeaponSystemEntity"] = edge(
        label="IS_A",
        description="Weapon system categories that this AAA system is a member of.",
        examples=[["Phalanx CIWS"], ["Bofors 40 mm L/70"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="ENGAGES",
        description="Platforms or target types that this AAA system can engage.",
        examples=[["MiG-29", "Cruise Missile"], ["Helicopter"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ElectronicWarfareSystemEntity(BaseModel):
    """Electronic Warfare System — EA, ES, or EP system (jammer, receiver, self-protection suite)
    """
    model_config = ConfigDict(ontology_name="ELECTRONIC_WARFARE_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the EW system", examples=['AN/ALQ-99', 'AN/ALQ-218'])
    nomenclature: Optional[str] = Field(default=None, description="Military AN/ designation", examples=['AN/ALQ-99'])
    ELNOT: Optional[str] = Field(default=None, description="Electronic Intelligence Notation identifier", examples=["FAN_SONG", "BIG_BIRD"])
    ew_role: Optional[str] = Field(default=None, description="Electronic warfare functional role", json_schema_extra={"enum": ["EA", "ES", "EP", "SIGINT", "ELINT", "COMINT"]})
    coverage: Optional[str] = Field(default=None, description="Frequency or angular coverage range", examples=['64 MHz - 40 GHz'])
    power_output: Optional[str] = Field(default=None, description="Maximum effective radiated power output", examples=['10 kW'])
    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this EW system is installed.",
        examples=["EA-18G Growler", "F-16CJ Block 50"],
        default=None,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class FireControlSystemEntity(BaseModel):
    """Fire Control System — System that provides targeting and engagement control
    """
    model_config = ConfigDict(ontology_name="FIRE_CONTROL_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the fire control system", examples=['AN/MPQ-65 Radar Set', 'AN/TPY-2 Radar'])
    nomenclature: Optional[str] = Field(default=None, description="Military AN/ designation", examples=['AN/MPQ-65'])
    missile_systems: List["MissileSystemEntity"] = edge(
        label="GUIDES",
        description="Missile systems that this fire control system guides to target.",
        examples=[["PAC-3 MSE", "SM-6 Block IA"], ["THAAD Interceptor"]],
        default_factory=list,
    )
    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this fire control system is installed.",
        examples=["SA-20 TEL", "Patriot PAC-3 ICC"],
        default=None,
    )
    platforms: List["PlatformEntity"] = edge(
        label="TRACKS",
        description="Platforms or targets that this fire control system tracks.",
        examples=[["MiG-29", "Ballistic Missile"], ["Cruise Missile"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="DESIGNATES",
        description="Platforms or targets that this fire control system designates for engagement.",
        examples=[["SA-20 TEL", "MiG-29"], ["Ballistic Missile"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class IntegratedAirDefenseSystemEntity(BaseModel):
    """Integrated Air Defense System — System-of-systems combining radar, missile, AAA, and C2 for area defense
    """
    model_config = ConfigDict(ontology_name="INTEGRATED_AIR_DEFENSE_SYSTEM", graph_id_fields=['name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    name: str = Field(..., description="Common or NATO reporting name of the IADS", examples=['S-400 Triumf', 'Patriot PAC-3'])
    status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED"]})
    doctrine: Optional[str] = Field(default=None, description="Employment doctrine or concept of operations", examples=['Layered defense with multi-range engagement'])
    radar_systems: List["RadarSystemEntity"] = edge(
        label="CONTAINS",
        description="Radar systems contained within this integrated air defense system.",
        examples=[["Tombstone", "Clam Shell"], ["AN/MPQ-65"]],
        default_factory=list,
    )
    missile_systems: List["MissileSystemEntity"] = edge(
        label="CONTAINS",
        description="Missile systems contained within this integrated air defense system.",
        examples=[["PAC-3 MSE", "SM-6 Block IA"], ["THAAD Interceptor"]],
        default_factory=list,
    )
    air_defense_artillery_systems: List["AirDefenseArtillerySystemEntity"] = edge(
        label="CONTAINS",
        description="AAA systems contained within this integrated air defense system.",
        examples=[["ZSU-23-4 Shilka", "Gepard FlakPanzer"], ["S-60 57mm"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="DEPLOYED_ON",
        description="Platforms on which this IADS is deployed.",
        examples=[["SA-20 TEL", "Patriot PAC-3 ICC"], ["Mobile Radar Site"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="SUPPORTS_ENGAGEMENT_OF",
        description="Platforms or target types that this IADS supports engagement of.",
        examples=[["MiG-29", "Ballistic Missile"], ["Cruise Missile"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class LauncherSystemEntity(BaseModel):
    """Launcher System — Missile or rocket launcher platform
    """
    model_config = ConfigDict(ontology_name="LAUNCHER_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the launcher system", examples=['M903 Launching Station', 'M270 MLRS'])
    launcher_type: Optional[str] = Field(default=None, description="Category of launcher mechanism", examples=['Vertical cold-launch canister'])
    capacity: Optional[int] = Field(default=None, description="Number of missiles the launcher can hold", examples=[16])
    missile_systems: List["MissileSystemEntity"] = edge(
        label="LAUNCHES",
        description="Missile systems that this launcher is capable of launching.",
        examples=[["PAC-3 MSE", "SM-6 Block IA"], ["THAAD Interceptor"]],
        default_factory=list,
    )
    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this launcher system is installed.",
        examples=["M903 Launching Station", "SA-20 TEL"],
        default=None,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})



# ----------------------------------------------------------------------
# ALL_ENTITIES registry
# ----------------------------------------------------------------------

ALL_ENTITIES: dict[str, type[BaseModel]] = {
    "DOCUMENT": DocumentEntity,
    "SECTION": SectionEntity,
    "FIGURE": FigureEntity,
    "TABLE": TableEntity,
    "IMAGE": ImageEntity,
    "TEXT_BLOCK": TextBlockEntity,
    "ORGANIZATION": OrganizationEntity,
    "PLATFORM": PlatformEntity,
    "WEAPON_SYSTEM": WeaponSystemEntity,
    "EQUIPMENT_SYSTEM": EquipmentSystemEntity,
    "SUBSYSTEM": SubsystemEntity,
    "COMPONENT": ComponentEntity,
    "RADAR_SYSTEM": RadarSystemEntity,
    "MISSILE_SYSTEM": MissileSystemEntity,
    "AIR_DEFENSE_ARTILLERY_SYSTEM": AirDefenseArtillerySystemEntity,
    "ELECTRONIC_WARFARE_SYSTEM": ElectronicWarfareSystemEntity,
    "FIRE_CONTROL_SYSTEM": FireControlSystemEntity,
    "INTEGRATED_AIR_DEFENSE_SYSTEM": IntegratedAirDefenseSystemEntity,
    "LAUNCHER_SYSTEM": LauncherSystemEntity,
}

# Rebuild forward references for all entity classes.
for _cls in ALL_ENTITIES.values():
    _cls.model_rebuild()
