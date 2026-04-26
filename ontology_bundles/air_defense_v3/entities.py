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
):
    """Field constructor for system_metadata (spec §3.3 bucket 2).

    Real, indexed field — never surfaced by a starter profile. Used
    for audit trails, classifier IDs, status flags, review cadence.
    """
    extra = {"profile_sections": [], "system_metadata": True}
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
):
    """Field constructor for identity adjuncts (spec §3.3 bucket 3).

    Used for fields like `nomenclature` and the missile-schema `name`
    that are identity context but not graph_id_fields. The four-bucket
    contract test treats them as identity rather than profile-mapped
    or metadata.
    """
    extra = {"profile_sections": [], "identity_field": True}
    kwargs: dict[str, Any] = {
        "default": default,
        "description": description,
        "json_schema_extra": extra,
    }
    if examples is not None:
        kwargs["examples"] = examples
    return Field(**kwargs)


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
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities that this equipment system provides.",
        examples=[["Terminal Phase Guidance", "Initial Target Acquisition"], ["Mid-Course Tracking"]],
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
    standards: List["StandardEntity"] = edge(
        label="SPECIFIED_BY",
        description="Military standards that specify requirements for this equipment system.",
        examples=[["MIL-STD-1553B", "MIL-DTL-31000G"], ["MIL-STD-810H"]],
        default_factory=list,
    )
    specifications: List["SpecificationEntity"] = edge(
        label="SPECIFIED_BY",
        description="Performance specifications that define requirements for this equipment system.",
        examples=[["max_range=150 km"], ["operating_temperature=-40 to +55 C"]],
        default_factory=list,
    )
    test_events: List["TestEventEntity"] = edge(
        label="TESTED_IN",
        description="Test or evaluation events in which this equipment system was tested.",
        examples=[["FET-10 Flight Test", "IOT&E Phase 2"], ["LFT&E Arena Test"]],
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
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities that this subsystem provides.",
        examples=[["Terminal Phase Guidance", "Initial Target Acquisition"], ["Mid-Course Tracking"]],
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
    standards: List["StandardEntity"] = edge(
        label="SPECIFIED_BY",
        description="Military standards that specify requirements for this component.",
        examples=[["MIL-STD-1553B", "MIL-DTL-31000G"], ["MIL-STD-810H"]],
        default_factory=list,
    )
    test_events: List["TestEventEntity"] = edge(
        label="TESTED_IN",
        description="Test or evaluation events in which this component was tested.",
        examples=[["FET-10 Flight Test"], ["IOT&E Phase 2", "LFT&E Arena Test"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RadarSystemEntity(BaseModel):
    """Radar System — Active or passive radar system (search, track, fire control, SAR, AESA, PESA)
    """
    model_config = ConfigDict(ontology_name="RADAR_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the radar system", examples=['Tombstone', 'Clam Shell'])

    # ===== Flat-checklist fields (spec §3.3) =====
    # Waveform group — RF Parameters only
    nominal_rf_mhz: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Nominal radio frequency in MHz at which the radar transmits.",
        examples=[3000.0, 9300.0],
    )
    frequency_excursion_mhz: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Total instantaneous frequency excursion during a coherent processing interval, in MHz.",
        examples=[5.0, 50.0],
    )
    nominal_pri_usec: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulse repetition interval in microseconds.",
        examples=[1000.0],
    )
    nominal_pd_usec: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulse duration in microseconds.",
        examples=[1.0],
    )
    inter_pulse: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Inter-pulse modulation pattern.",
        examples=["staggered PRI", "fixed"],
    )
    pulses_per_dwell: Optional[int] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulses per coherent dwell.",
        examples=[16],
    )
    dwell_time: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Coherent dwell time, seconds.",
        examples=[0.05],
    )
    intra_pulse_mop: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Intra-pulse modulation on pulse.",
        examples=["LFM", "BPSK"],
    )
    num_bits_in_code: Optional[int] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Number of bits in the phase-code (when phase-coded MOP).",
        examples=[13],
    )

    # Antenna group — RF Parameters and Components
    antenna_dim_az_m: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna aperture, azimuth dimension, meters.",
        examples=[6.0],
    )
    antenna_dim_el_m: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna aperture, elevation dimension, meters.",
        examples=[2.0],
    )
    beamwidth_az_deg: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="One-way 3 dB azimuth beamwidth, degrees.",
        examples=[1.5],
    )
    beamwidth_el_deg: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="One-way 3 dB elevation beamwidth, degrees.",
        examples=[2.0],
    )
    gain_dbi: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna gain, dBi.",
        examples=[35.0],
    )
    antenna_photo: Optional[bool] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Whether a photo of the antenna is available in source documents.",
    )
    spoiled: Optional[bool] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Whether the antenna pattern is spoiled (broadened) for surveillance.",
    )
    coverage_limits_el_deg: Optional[str] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Elevation coverage limits, degrees (e.g. '0–60').",
        examples=["0–60"],
    )

    # Transmit group — RF Parameters and Performance
    tx_peak_power_kw: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="transmit",
        description="Transmitter peak power, kilowatts.",
        examples=[600.0],
    )
    erp_dbw: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="transmit",
        description="Effective radiated power, dBW.",
        examples=[88.0],
    )

    # Scan group — RF Parameters and Performance
    scan_type: Optional[str] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="scan",
        description="Scan mechanism / mode (e.g. 'mechanical', 'phased-array', 'electronic').",
        examples=["phased-array"],
    )
    scan_period_sec: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="scan",
        description="Scan revisit / repeat period, seconds.",
        examples=[10.0],
    )

    # Classification group — RF Parameters
    emitter_function: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="classification",
        description="Primary emitter function (e.g. 'acquisition', 'tracking', 'engagement').",
        examples=["tracking"],
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        description=(
            "Official military nomenclature — formal alphanumeric designator "
            "(JETDS for US, GRAU index for Russian). Distinct from system_name."
        ),
        examples=["AN/MPQ-65", "5N63S", "30N6E"],
    )

    # System metadata
    elnot: Optional[str] = metadata_field(
        description="Emitter library number (ELNOT) — IC enumeration.",
        examples=["E0123"],
    )
    dieqp: Optional[str] = metadata_field(
        description="Digital Intelligence Equipment Parameters cross-reference identifier.",
    )
    asrd: Optional[str] = metadata_field(
        description="ASRD identifier — IC source-of-record reference.",
    )
    system_status: Optional[str] = metadata_field(
        description="Operational status (e.g. 'in service', 'retired', 'in development').",
        examples=["in service"],
    )
    responsible_agency: Optional[str] = metadata_field(
        description="Agency or organization that owns the parametric record.",
        examples=["NASIC"],
    )
    review_cycle: Optional[str] = metadata_field(
        description="Review cadence for the parametric record.",
        examples=["annual"],
    )
    next_review_date: Optional[str] = metadata_field(
        description="Next scheduled review date for the record (YYYY-MM-DD).",
        examples=["2027-04-01"],
    )

    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this radar system is installed.",
        examples=["SA-20 TEL", "Patriot PAC-3 ICC"],
        default=None,
    )
    waveforms: List["WaveformEntity"] = edge(
        label="USES_WAVEFORM",
        description="Waveforms used by this radar system for transmission.",
        examples=[["Search Mode 1", "Track Mode 3"], ["Burst Mode 2"]],
        default_factory=list,
    )
    rf_emissions: List["RfEmissionEntity"] = edge(
        label="EMITS",
        description="RF emissions generated by this radar system.",
        examples=[["Tombstone Search Mode"], ["Clam Shell Track Mode"]],
        default_factory=list,
    )
    antennas: List["AntennaEntity"] = edge(
        label="HAS_ANTENNA",
        description="Antennas that are part of this radar system.",
        examples=[["Main Array Antenna", "IFF Antenna"], ["Search Antenna"]],
        default_factory=list,
    )
    receivers: List["ReceiverEntity"] = edge(
        label="HAS_RECEIVER",
        description="Receiver subsystems that are part of this radar system.",
        examples=[["Main Receiver Unit", "Auxiliary Receiver Unit"], ["Digital Receiver"]],
        default_factory=list,
    )
    transmitters: List["TransmitterEntity"] = edge(
        label="HAS_TRANSMITTER",
        description="Transmitter subsystems that are part of this radar system.",
        examples=[["Main Transmitter Unit"], ["Backup Transmitter Unit"]],
        default_factory=list,
    )
    scan_patterns: List["ScanPatternEntity"] = edge(
        label="HAS_SCAN",
        description="Antenna scan patterns used by this radar system.",
        examples=[["Conical scan"], ["Raster scan", "Sector scan"]],
        default_factory=list,
    )
    signal_processing_chains: List["SignalProcessingChainEntity"] = edge(
        label="HAS_PROCESSING_CHAIN",
        description="Signal processing chains associated with this radar system.",
        examples=[["Main Processing Chain", "MTI Filter Chain"], ["Doppler Filter Chain"]],
        default_factory=list,
    )
    radar_performances: List["RadarPerformanceEntity"] = edge(
        label="HAS_PERFORMANCE",
        description="Performance characteristics envelope for this radar system.",
        examples=[["max_detection_range_1sqm_km=300"], ["max_unambiguous_range_km=400"]],
        default_factory=list,
    )
    frequency_bands: List["FrequencyBandEntity"] = edge(
        label="OPERATES_IN_BAND",
        description="Frequency bands in which this radar system operates.",
        examples=[["X-band", "S-band"], ["Ku-band"]],
        default_factory=list,
    )
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities provided by this radar system.",
        examples=[["Terminal Phase Guidance", "Initial Target Acquisition"], ["Mid-Course Tracking"]],
        default_factory=list,
    )
    rf_signatures: List["RfSignatureEntity"] = edge(
        label="HAS_SIGNATURE",
        description="RF signatures associated with this radar system for ELINT identification.",
        examples=[["Tombstone Track Signature"], ["Clam Shell Search Signature"]],
        default_factory=list,
    )
    engagement_timelines: List["EngagementTimelineEntity"] = edge(
        label="HAS_TIMELINE",
        description="Engagement timelines associated with this radar system.",
        examples=[["detection_to_designate_time_s=4"], ["designation_to_launch_time_s=8"]],
        default_factory=list,
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
    specifications: List["SpecificationEntity"] = edge(
        label="SPECIFIED_BY",
        description="Performance specifications that define requirements for this radar system.",
        examples=[["max_range=300 km"], ["operating_temperature=-40 to +55 C"]],
        default_factory=list,
    )
    test_events: List["TestEventEntity"] = edge(
        label="TESTED_IN",
        description="Test or evaluation events in which this radar system was tested.",
        examples=[["FET-10 Flight Test", "IOT&E Phase 2"], ["LFT&E Arena Test"]],
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
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class MissileSystemEntity(BaseModel):
    """Missile System — Guided missile weapon system with seeker, guidance, and propulsion
    """
    model_config = ConfigDict(ontology_name="MISSILE_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the missile system", examples=['PAC-3 MSE', 'SM-6 Block IA'])

    # ===== Flat-checklist fields (spec §3.3) =====
    # Airframe group — Components + Performance
    body_length_m: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Missile body length, meters.",
        examples=[10.6],
    )
    body_diameter_m: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Missile body diameter, meters.",
        examples=[0.5],
    )
    total_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Total missile mass at launch, kilograms.",
        examples=[2300.0],
    )
    missile_photo: Optional[bool] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Whether a photo of the missile is available in source documents.",
    )

    # Seeker group — Components + Performance
    seeker_type: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="seeker",
        description="Seeker type (e.g. 'semi-active radar', 'IR', 'inertial+command').",
        examples=["semi-active radar"],
    )

    # Booster group — Components + Performance
    booster_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster burn duration, seconds.",
        examples=[5.0],
    )
    booster_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster thrust (string — units may vary in source documents).",
        examples=["50 kN"],
    )
    booster_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster section mass, kilograms.",
        examples=[200.0],
    )

    # Sustain group — Components + Performance
    sustain_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor burn duration, seconds.",
        examples=[60.0],
    )
    sustain_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor thrust (string — units may vary).",
        examples=["10 kN"],
    )
    sustain_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor section mass, kilograms.",
        examples=[100.0],
    )

    # Ejector group — Components + Performance
    ejector_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge burn duration, seconds.",
        examples=[0.2],
    )
    ejector_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge thrust (string — units may vary).",
    )
    ejector_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge mass, kilograms.",
    )

    # Engagement envelope — Performance only
    min_intercept_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Minimum intercept range, kilometers.",
        examples=[3.0],
    )
    max_intercept_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum intercept range, kilometers.",
        examples=[150.0],
    )
    min_altitude_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Minimum engagement altitude, kilometers.",
        examples=[0.05],
    )
    max_altitude_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum engagement altitude, kilometers.",
        examples=[30.0],
    )
    max_launch_angle_deg: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum launch elevation angle from vertical, degrees.",
        examples=[60.0],
    )

    # Kinematics — Performance only
    average_speed_mps: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Average flight speed, meters per second.",
    )
    max_speed_mps: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Maximum flight speed, meters per second.",
        examples=[1100.0],
    )
    max_flyout_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Maximum flyout duration to engagement, seconds.",
    )
    flight_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Nominal flight duration, seconds.",
    )
    coast_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Coast (unpowered) phase duration, seconds.",
    )
    total_burn_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Total powered burn duration across all motors, seconds.",
    )
    intra_salvo_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Inter-shot interval within a salvo, seconds.",
    )

    # Guidance — Performance only
    guidance_type: Optional[str] = profile_field(
        sections=["performance"], subgroup="guidance",
        description="Guidance approach (e.g. 'command', 'inertial+command+terminal-active').",
        examples=["command + terminal-SARH"],
    )

    # Classification — Performance only
    emitter_function: Optional[str] = profile_field(
        sections=["performance"], subgroup="classification",
        description="Primary emitter function for the missile's seeker / data link.",
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        description="Military designation or NATO reporting name.",
        examples=["MIM-104F"],
    )
    name: Optional[str] = identity_field(
        description=(
            "Secondary alias / common name. The missile schema's secondary "
            "alias field; rendered after nomenclature on the entity header."
        ),
    )

    # System metadata
    dieqp: Optional[str] = metadata_field(
        description="Digital Intelligence Equipment Parameters cross-reference identifier.",
    )
    asrd: Optional[str] = metadata_field(
        description="ASRD identifier — IC source-of-record reference.",
    )
    system_status: Optional[str] = metadata_field(
        description="Operational status.",
        examples=["in service"],
    )
    responsible_agency: Optional[str] = metadata_field(
        description="Agency or organization that owns the parametric record.",
    )
    review_cycle: Optional[str] = metadata_field(
        description="Review cadence for the parametric record.",
    )
    next_review_date: Optional[str] = metadata_field(
        description="Next scheduled review date for the record (YYYY-MM-DD).",
    )

    guidance_method: Optional["GuidanceMethodEntity"] = edge(
        label="HAS_GUIDANCE",
        description="Guidance method used by this missile system.",
        examples=["Active radar homing", "Command guidance"],
        default=None,
    )
    seeker: Optional["SeekerEntity"] = edge(
        label="HAS_SEEKER",
        description="Terminal guidance seeker head used by this missile system.",
        examples=["Ka-band active seeker", "Ku-band semi-active seeker"],
        default=None,
    )
    propulsion_stacks: List["PropulsionStackEntity"] = edge(
        label="HAS_PROPULSION",
        description="Propulsion stacks that provide thrust for this missile system.",
        examples=[["total_burntime_s=25"], ["total_burntime_s=18"]],
        default_factory=list,
    )
    missile_performances: List["MissilePerformanceEntity"] = edge(
        label="HAS_PERFORMANCE",
        description="Performance characteristics envelope for this missile system.",
        examples=[["maximum_range_km=160"], ["maximum_altitude_km=25"]],
        default_factory=list,
    )
    platform: Optional["PlatformEntity"] = edge(
        label="INSTALLED_ON",
        description="Platform on which this missile system is installed.",
        examples=["M903 Launching Station", "SA-20 TEL"],
        default=None,
    )
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities provided by this missile system.",
        examples=[["Terminal Phase Guidance", "Initial Target Acquisition"], ["Mid-Course Tracking"]],
        default_factory=list,
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
    specifications: List["SpecificationEntity"] = edge(
        label="SPECIFIED_BY",
        description="Performance specifications that define requirements for this missile system.",
        examples=[["max_range=160 km"], ["maximum_altitude=25 km"]],
        default_factory=list,
    )
    test_events: List["TestEventEntity"] = edge(
        label="TESTED_IN",
        description="Test or evaluation events in which this missile system was tested.",
        examples=[["FET-10 Flight Test", "IOT&E Phase 2"], ["LFT&E Arena Test"]],
        default_factory=list,
    )
    platforms: List["PlatformEntity"] = edge(
        label="ENGAGES",
        description="Platforms or target types that this missile system can engage.",
        examples=[["MiG-29", "Cruise Missile"], ["Ballistic Missile"]],
        default_factory=list,
    )
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

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
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities provided by this AAA system.",
        examples=[["Air Defense", "Short-Range Engagement"], ["Anti-Aircraft Coverage"]],
        default_factory=list,
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
    frequency_bands: List["FrequencyBandEntity"] = edge(
        label="OPERATES_IN_BAND",
        description="Frequency bands in which this EW system operates.",
        examples=[["X-band", "S-band"], ["Ku-band", "Ka-band"]],
        default_factory=list,
    )
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities provided by this EW system.",
        examples=[["Electronic Attack", "Electronic Support"], ["SIGINT Collection"]],
        default_factory=list,
    )
    rf_emissions: List["RfEmissionEntity"] = edge(
        label="DETECTS",
        description="RF emissions that this EW system can detect.",
        examples=[["Tombstone Search Mode"], ["Clam Shell Track Mode"]],
        default_factory=list,
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
    capabilities: List["CapabilityEntity"] = edge(
        label="PROVIDES",
        description="Operational capabilities provided by this integrated air defense system.",
        examples=[["Air Defense", "Layered Engagement"], ["Area Defense"]],
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
