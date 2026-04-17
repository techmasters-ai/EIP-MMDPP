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


# ----------------------------------------------------------------------
# Layer 1
# ----------------------------------------------------------------------

class DocumentEntity(BaseModel):
    """Source Document — Technical manual, specification, report, drawing, or other source artifact
    """
    model_config = ConfigDict(ontology_name="DOCUMENT", graph_id_fields=["document_number"], identity_scope="global", dodaf_parent="DocumentResource", is_entity=True)

    document_number: str = Field(..., description="Official document designator (TM/MIL-STD/MIL-DTL number)", examples=['TM 9-1425-386-12', 'MIL-STD-1553B', 'MIL-DTL-31000G'])
    title: Optional[str] = Field(default=None, description="Full title of the document", examples=['Operator Manual for Patriot Missile System'])
    document_id: Optional[str] = Field(default=None, description="Internal system-assigned document identifier")
    classification: Optional[str] = Field(default=None, description="Security classification level", examples=['UNCLASSIFIED'])
    publication_date: Optional[str] = Field(default=None, description="Publication or revision date (YYYY-MM-DD)", examples=['2023-06-15'])
    source_type: Optional[str] = Field(default=None, description="Category of the source document", json_schema_extra={"enum": ["MANUAL", "REPORT", "BRIEFING", "NOTE", "SCHEMATIC", "DRAWING", "SPECIFICATION", "CHECKLIST", "SPREADSHEET"]})
    issuing_org: Optional[str] = Field(default=None, description="Organization that published the document", examples=['U.S. Army TACOM'])
    language: Optional[str] = Field(default=None, description="Language of the document (ISO 639-1 code)", examples=['en'])
    workbook_name: Optional[str] = Field(default=None, description="Name of the spreadsheet workbook file", examples=['SA-20_MDE_Checklist.xlsx'])
    sheet_name: Optional[str] = Field(default=None, description="Name of the specific worksheet tab", examples=['Radar Parameters'])
    documents: List["DocumentEntity"] = edge(label="DERIVED_FROM", default_factory=list)
    documents: List["DocumentEntity"] = edge(label="SUPERSEDES", default_factory=list)
    organizations: List["OrganizationEntity"] = edge(label="REVIEWED_BY", default_factory=list)
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

    name: str = Field(..., description="Common name of the platform", examples=['SA-20 TEL', 'SA-20 TEL'])
    platform_designation: Optional[str] = Field(default=None, description="Military designation of the platform", examples=['5P85SE'])
    platform_type: Optional[str] = Field(default=None, description="Category of the platform", json_schema_extra={"enum": ["AIRCRAFT", "SHIP", "GROUND_VEHICLE", "FIXED_SITE", "SPACE", "MOBILE_LAUNCHER", "AIR_DEFENSE_BATTERY"]})
    country: Optional[str] = Field(default=None, description="Country of origin or primary operator", examples=['Russia'])
    service_branch: Optional[str] = Field(default=None, description="Military branch operating the platform", examples=['Russian Aerospace Forces'])
    platform_status: Optional[str] = Field(default=None, description="Current lifecycle status of the platform", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    organizations: List["OrganizationEntity"] = edge(label="OPERATED_BY", default_factory=list)
    organizations: List["OrganizationEntity"] = edge(label="MANUFACTURED_BY", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="INSTANCE_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class WeaponSystemEntity(BaseModel):
    """Weapon System — Generic weapon system (not radar, missile, or AAA specifically)
    """
    model_config = ConfigDict(ontology_name="WEAPON_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the weapon system", examples=['Phalanx CIWS', 'Phalanx CIWS'])
    nomenclature: Optional[str] = Field(default=None, description="Military designation", examples=['Mk 15'])
    weapon_type: Optional[str] = Field(default=None, description="Category of weapon system", examples=['Close-In Weapon System'])
    subsystems: List["SubsystemEntity"] = edge(label="CONTAINS", default_factory=list)
    subsystems: List["SubsystemEntity"] = edge(label="HAS_SUBSYSTEM", default_factory=list)
    components: List["ComponentEntity"] = edge(label="HAS_COMPONENT", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="ENGAGES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class EquipmentSystemEntity(BaseModel):
    """Equipment System — Top-level integrated weapon or defense system (generic)
    """
    model_config = ConfigDict(ontology_name="EQUIPMENT_SYSTEM", graph_id_fields=[], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    name: Optional[str] = Field(default=None, description="Common name or designation of the system", examples=['Patriot PAC-3'])
    designation: Optional[str] = Field(default=None, description="Military AN/ designation or program designation", examples=['AN/MPQ-65'], pattern='^AN/[A-Z]{3}-\\d+')
    program_office: Optional[str] = Field(default=None, description="Managing program office", examples=['PEO Missiles and Space'])
    status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    prime_contractor: Optional[str] = Field(default=None, description="Lead contractor organization", examples=['Lockheed Martin'])
    service_branch: Optional[str] = Field(default=None, description="Military branch operating the system", examples=['U.S. Army'])
    components: List["ComponentEntity"] = edge(label="CONTAINS", default_factory=list)
    subsystems: List["SubsystemEntity"] = edge(label="HAS_SUBSYSTEM", default_factory=list)
    components: List["ComponentEntity"] = edge(label="HAS_COMPONENT", default_factory=list)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    organizations: List["OrganizationEntity"] = edge(label="MANUFACTURED_BY", default_factory=list)
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="INSTANCE_OF", default_factory=list)
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="ALIAS_OF", default_factory=list)
    standards: List["StandardEntity"] = edge(label="SPECIFIED_BY", default_factory=list)
    specifications: List["SpecificationEntity"] = edge(label="SPECIFIED_BY", default_factory=list)
    test_events: List["TestEventEntity"] = edge(label="TESTED_IN", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SubsystemEntity(BaseModel):
    """Subsystem — Major functional component within a military system
    """
    model_config = ConfigDict(ontology_name="SUBSYSTEM", graph_id_fields=[], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name of the subsystem", examples=['Guidance Section'])
    subsystem_role: Optional[str] = Field(default=None, description="Functional role within the parent system", examples=['Signal processing and target tracking'])
    part_number: Optional[str] = Field(default=None, description="Subsystem-level part or drawing number", examples=['GS-PAC3-001'])
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="PART_OF", default_factory=list)
    components: List["ComponentEntity"] = edge(label="HAS_COMPONENT", default_factory=list)
    radar_systems: List["RadarSystemEntity"] = edge(label="PART_OF", default_factory=list)
    missile_systems: List["MissileSystemEntity"] = edge(label="PART_OF", default_factory=list)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ComponentEntity(BaseModel):
    """Component — Individual physical or logical part within a subsystem or assembly
    """
    model_config = ConfigDict(ontology_name="COMPONENT", graph_id_fields=[], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    name: Optional[str] = Field(default=None, description="Common name of the component", examples=['Traveling Wave Tube'])
    component_type: Optional[str] = Field(default=None, description="Category or class of the component", examples=['Amplifier'])
    part_number: Optional[str] = Field(default=None, description="Manufacturer-assigned part number (alphanumeric with dashes)", examples=['PN-12345-A'], pattern='^[A-Z0-9][A-Z0-9\\-/]{2,20}$')
    nsn: Optional[str] = Field(default=None, description="National Stock Number in NNNN-NN-NNN-NNNN format", examples=['5961-01-234-5678'], pattern='^\\d{4}-\\d{2}-\\d{3}-\\d{4}$')
    cage_code: Optional[str] = Field(default=None, description="5-character Commercial and Government Entity code identifying the manufacturer", examples=['1ABC3'], pattern='^[A-Z0-9]{5}$')
    manufacturer: Optional[str] = Field(default=None, description="Name of the component manufacturer", examples=['L3Harris Technologies'])
    material: Optional[str] = Field(default=None, description="Primary material composition of the component", examples=['Aluminum 7075-T6'])
    weight_kg: Optional[float] = Field(default=None, description="Weight of the component in kilograms", examples=[2.5])
    subsystems: List["SubsystemEntity"] = edge(label="PART_OF", default_factory=list)
    organizations: List["OrganizationEntity"] = edge(label="MANUFACTURED_BY", default_factory=list)
    standards: List["StandardEntity"] = edge(label="SPECIFIED_BY", default_factory=list)
    test_events: List["TestEventEntity"] = edge(label="TESTED_IN", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RadarSystemEntity(BaseModel):
    """Radar System — Active or passive radar system (search, track, fire control, SAR, AESA, PESA)
    """
    model_config = ConfigDict(ontology_name="RADAR_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the radar system", examples=['Tombstone', 'Tombstone'])
    nomenclature: Optional[str] = Field(default=None, description="Military AN/ or NATO reporting nomenclature", examples=['AN/MPQ-65'])
    ELNOT: Optional[str] = Field(default=None, description="Electronic Intelligence Notation identifier", examples=['TOMBSTONE'])
    DIEQP: Optional[str] = Field(default=None, description="Defense Intelligence Equipment identifier code")
    radar_type: Optional[str] = Field(default=None, description="Functional category of the radar", json_schema_extra={"enum": ["SEARCH", "FIRE_CONTROL", "TRACKING", "WEATHER", "SAR", "GMTI", "MULTI_FUNCTION", "AESA", "PESA", "MPAR"]})
    emitter_function: Optional[str] = Field(default=None, description="Primary emitter function or role", examples=['Surveillance and target acquisition'])
    system_status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    responsible_agency: Optional[str] = Field(default=None, description="Organization responsible for system management", examples=['PEO IEW&S'])
    nominal_frequency: Optional[str] = Field(default=None, description="Nominal operating frequency or center frequency", examples=['9.4 GHz'])
    frequency_limits: Optional[str] = Field(default=None, description="Frequency range limits (min-max)", examples=['8.5-10.5 GHz'])
    radar_waveform: Optional[str] = Field(default=None, description="Primary waveform type used", examples=['Pulse Doppler LFM chirp'])
    nominal_PRI: Optional[str] = Field(default=None, description="Nominal pulse repetition interval", examples=['500 us'])
    PRI_limits: Optional[str] = Field(default=None, description="PRI range limits (min-max)", examples=['200-1000 us'])
    PRF_limits: Optional[str] = Field(default=None, description="Pulse repetition frequency range limits", examples=['1000-5000 Hz'])
    nominal_pulse_duration: Optional[str] = Field(default=None, description="Nominal pulse width or duration", examples=['10 us'])
    pulse_duration_limits: Optional[str] = Field(default=None, description="Pulse duration range limits (min-max)", examples=['1-100 us'])
    ERP: Optional[str] = Field(default=None, description="Effective radiated power", examples=['1.2 MW'])
    tx_peak_power: Optional[str] = Field(default=None, description="Transmitter peak power output", examples=['150 kW'])
    duty_cycle: Optional[float] = Field(default=None, description="Transmitter duty cycle (0.0 to 1.0)", examples=[0.05])
    gain: Optional[float] = Field(default=None, description="Antenna gain in dBi", examples=[38.0])
    scan_type: Optional[str] = Field(default=None, description="Antenna scan mode", examples=['Electronic beam steering'])
    scan_period: Optional[str] = Field(default=None, description="Time for one complete scan cycle", examples=['6 s'])
    detection_to_designate_time: Optional[str] = Field(default=None, description="Time from detection to target designation", examples=['4 s'])
    designation_to_launch_time: Optional[str] = Field(default=None, description="Time from designation to missile launch", examples=['8 s'])
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    waveforms: List["WaveformEntity"] = edge(label="USES_WAVEFORM", default_factory=list)
    rf_emissions: List["RfEmissionEntity"] = edge(label="EMITS", default_factory=list)
    antennas: List["AntennaEntity"] = edge(label="HAS_ANTENNA", default_factory=list)
    receivers: List["ReceiverEntity"] = edge(label="HAS_RECEIVER", default_factory=list)
    transmitters: List["TransmitterEntity"] = edge(label="HAS_TRANSMITTER", default_factory=list)
    scan_patterns: List["ScanPatternEntity"] = edge(label="HAS_SCAN", default_factory=list)
    signal_processing_chains: List["SignalProcessingChainEntity"] = edge(label="HAS_PROCESSING_CHAIN", default_factory=list)
    radar_performances: List["RadarPerformanceEntity"] = edge(label="HAS_PERFORMANCE", default_factory=list)
    frequency_bands: List["FrequencyBandEntity"] = edge(label="OPERATES_IN_BAND", default_factory=list)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    rf_signatures: List["RfSignatureEntity"] = edge(label="HAS_SIGNATURE", default_factory=list)
    engagement_timelines: List["EngagementTimelineEntity"] = edge(label="HAS_TIMELINE", default_factory=list)
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="IS_A", default_factory=list)
    radar_systems: List["RadarSystemEntity"] = edge(label="ALIAS_OF", default_factory=list)
    specifications: List["SpecificationEntity"] = edge(label="SPECIFIED_BY", default_factory=list)
    test_events: List["TestEventEntity"] = edge(label="TESTED_IN", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="TRACKS", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="DETECTS", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="DESIGNATES", default_factory=list)
    missile_systems: List["MissileSystemEntity"] = edge(label="SUPPORTS_ENGAGEMENT_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class MissileSystemEntity(BaseModel):
    """Missile System — Guided missile weapon system with seeker, guidance, and propulsion
    """
    model_config = ConfigDict(ontology_name="MISSILE_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the missile system", examples=['PAC-3 MSE', 'PAC-3 MSE'])
    nomenclature: Optional[str] = Field(default=None, description="Military designation or NATO reporting name", examples=['MIM-104F'])
    DIEQP: Optional[str] = Field(default=None, description="Defense Intelligence Equipment identifier code")
    system_status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED", "PROTOTYPE"]})
    guidance_type: Optional[str] = Field(default=None, description="Primary guidance method used", examples=['Active radar homing'])
    seeker_nomenclature: Optional[str] = Field(default=None, description="Designation of the terminal guidance seeker", examples=['Ka-band active seeker'])
    seeker_ELNOT: Optional[str] = Field(default=None, description="ELNOT identifier for the seeker emitter")
    seeker_DIEQP: Optional[str] = Field(default=None, description="DIEQP code for the seeker")
    guidance_method: Optional["GuidanceMethodEntity"] = edge(label="HAS_GUIDANCE", default=None)
    seeker: Optional["SeekerEntity"] = edge(label="HAS_SEEKER", default=None)
    propulsion_stacks: List["PropulsionStackEntity"] = edge(label="HAS_PROPULSION", default_factory=list)
    missile_performances: List["MissilePerformanceEntity"] = edge(label="HAS_PERFORMANCE", default_factory=list)
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="DEFENDS", default_factory=list)
    weapon_systems: List["WeaponSystemEntity"] = edge(label="IS_A", default_factory=list)
    missile_systems: List["MissileSystemEntity"] = edge(label="ALIAS_OF", default_factory=list)
    specifications: List["SpecificationEntity"] = edge(label="SPECIFIED_BY", default_factory=list)
    test_events: List["TestEventEntity"] = edge(label="TESTED_IN", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="ENGAGES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class AirDefenseArtillerySystemEntity(BaseModel):
    """Air Defense Artillery System — Gun-based air defense system (AAA)
    """
    model_config = ConfigDict(ontology_name="AIR_DEFENSE_ARTILLERY_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the AAA system", examples=['ZSU-23-4 Shilka', 'ZSU-23-4 Shilka'])
    DIEQP: Optional[str] = Field(default=None, description="Defense Intelligence Equipment identifier code")
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
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    weapon_systems: List["WeaponSystemEntity"] = edge(label="IS_A", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="ENGAGES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ElectronicWarfareSystemEntity(BaseModel):
    """Electronic Warfare System — EA, ES, or EP system (jammer, receiver, self-protection suite)
    """
    model_config = ConfigDict(ontology_name="ELECTRONIC_WARFARE_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the EW system", examples=['AN/ALQ-99', 'AN/ALQ-99'])
    nomenclature: Optional[str] = Field(default=None, description="Military AN/ designation", examples=['AN/ALQ-99'])
    ELNOT: Optional[str] = Field(default=None, description="Electronic Intelligence Notation identifier")
    ew_role: Optional[str] = Field(default=None, description="Electronic warfare functional role", json_schema_extra={"enum": ["EA", "ES", "EP", "SIGINT", "ELINT", "COMINT"]})
    coverage: Optional[str] = Field(default=None, description="Frequency or angular coverage range", examples=['64 MHz - 40 GHz'])
    power_output: Optional[str] = Field(default=None, description="Maximum effective radiated power output", examples=['10 kW'])
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    frequency_bands: List["FrequencyBandEntity"] = edge(label="OPERATES_IN_BAND", default_factory=list)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    rf_emissions: List["RfEmissionEntity"] = edge(label="DETECTS", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class FireControlSystemEntity(BaseModel):
    """Fire Control System — System that provides targeting and engagement control
    """
    model_config = ConfigDict(ontology_name="FIRE_CONTROL_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the fire control system", examples=['AN/MPQ-65 Radar Set', 'AN/MPQ-65 Radar Set'])
    nomenclature: Optional[str] = Field(default=None, description="Military AN/ designation", examples=['AN/MPQ-65'])
    missile_systems: List["MissileSystemEntity"] = edge(label="GUIDES", default_factory=list)
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    platforms: List["PlatformEntity"] = edge(label="TRACKS", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="DESIGNATES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class IntegratedAirDefenseSystemEntity(BaseModel):
    """Integrated Air Defense System — System-of-systems combining radar, missile, AAA, and C2 for area defense
    """
    model_config = ConfigDict(ontology_name="INTEGRATED_AIR_DEFENSE_SYSTEM", graph_id_fields=['name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    name: str = Field(..., description="Common or NATO reporting name of the IADS", examples=['S-400 Triumf', 'S-400 Triumf'])
    status: Optional[str] = Field(default=None, description="Current lifecycle status", json_schema_extra={"enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED"]})
    doctrine: Optional[str] = Field(default=None, description="Employment doctrine or concept of operations", examples=['Layered defense with multi-range engagement'])
    radar_systems: List["RadarSystemEntity"] = edge(label="CONTAINS", default_factory=list)
    missile_systems: List["MissileSystemEntity"] = edge(label="CONTAINS", default_factory=list)
    air_defense_artillery_systems: List["AirDefenseArtillerySystemEntity"] = edge(label="CONTAINS", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="DEPLOYED_ON", default_factory=list)
    capabilities: List["CapabilityEntity"] = edge(label="PROVIDES", default_factory=list)
    platforms: List["PlatformEntity"] = edge(label="SUPPORTS_ENGAGEMENT_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class LauncherSystemEntity(BaseModel):
    """Launcher System — Missile or rocket launcher platform
    """
    model_config = ConfigDict(ontology_name="LAUNCHER_SYSTEM", graph_id_fields=['system_name'], identity_scope="global", dodaf_parent="MilitarySystem", is_entity=True)

    system_name: str = Field(..., description="Common name of the launcher system", examples=['M903 Launching Station', 'M903 Launching Station'])
    launcher_type: Optional[str] = Field(default=None, description="Category of launcher mechanism", examples=['Vertical cold-launch canister'])
    capacity: Optional[int] = Field(default=None, description="Number of missiles the launcher can hold", examples=[16])
    missile_systems: List["MissileSystemEntity"] = edge(label="LAUNCHES", default_factory=list)
    platform: Optional["PlatformEntity"] = edge(label="INSTALLED_ON", default=None)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


# ----------------------------------------------------------------------
# Layer 3
# ----------------------------------------------------------------------

class FrequencyBandEntity(BaseModel):
    """Frequency Band — IEEE/NATO frequency band designation
    """
    model_config = ConfigDict(ontology_name="FREQUENCY_BAND", graph_id_fields=['band_name'], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    band_name: str = Field(..., description="Common name of the frequency band", examples=['X-band', 'X-band'])
    designation: Optional[str] = Field(default=None, description="IEEE or NATO band letter designation", json_schema_extra={"enum": ["HF", "VHF", "UHF", "L", "S", "C", "X", "Ku", "K", "Ka", "V", "W", "mm"]})
    freq_min_mhz: Optional[float] = Field(default=None, description="Lower frequency limit of the band in MHz", examples=[8000])
    freq_max_mhz: Optional[float] = Field(default=None, description="Upper frequency limit of the band in MHz", examples=[12000])
    standard_family: Optional[str] = Field(default=None, description="Standard body that defined the band (IEEE or NATO)", examples=['IEEE'])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ModulationEntity(BaseModel):
    """Modulation — Intra-pulse or inter-pulse modulation characteristics
    """
    model_config = ConfigDict(ontology_name="MODULATION", graph_id_fields=[], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name or identifier of the modulation scheme", examples=['LFM Up-Chirp'])
    intra_pulse_modulation: Optional[str] = Field(default=None, description="Type of modulation within each pulse", examples=['Linear FM (up-chirp)'])
    inter_pulse_modulation: Optional[str] = Field(default=None, description="Type of modulation between pulses", examples=['Stagger 4-position'])
    frequency_excursion: Optional[str] = Field(default=None, description="Frequency deviation or chirp bandwidth", examples=['5 MHz'])
    code_bits: Optional[int] = Field(default=None, description="Number of bits in phase or frequency code", examples=[13])
    pulse_compression_ratio: Optional[float] = Field(default=None, description="Ratio of uncompressed to compressed pulse width", examples=[100])
    pulse_compression_gain_db: Optional[float] = Field(default=None, description="Processing gain from pulse compression in dB", examples=[20])
    pulse_compression_weighting_function: Optional[str] = Field(default=None, description="Weighting function applied to reduce sidelobes", examples=['Hamming'])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RfSignatureEntity(BaseModel):
    """RF Signature — Composite RF fingerprint for emitter identification
    """
    model_config = ConfigDict(ontology_name="RF_SIGNATURE", graph_id_fields=[], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name or identifier for the RF signature", examples=['Tombstone Track Signature'])
    nominal_RF: Optional[str] = Field(default=None, description="Nominal radio frequency", examples=['9.4 GHz'])
    nominal_PRI: Optional[str] = Field(default=None, description="Nominal pulse repetition interval", examples=['500 us'])
    nominal_PD: Optional[str] = Field(default=None, description="Nominal pulse duration", examples=['10 us'])
    scan_type: Optional[str] = Field(default=None, description="Observed antenna scan type", examples=['Sector scan'])
    scan_period: Optional[str] = Field(default=None, description="Observed scan period", examples=['3 s'])
    modulation_pattern: Optional[str] = Field(default=None, description="Observed modulation pattern description", examples=['LFM chirp with 4-position PRI stagger'])
    frequency_agility: Optional[str] = Field(default=None, description="Frequency agility behavior observed", examples=['Burst-to-burst random within 200 MHz'])
    beam_characteristics: Optional[str] = Field(default=None, description="Observed beam shape and scanning characteristics", examples=['Pencil beam, 1.5 deg azimuth'])
    dwell_time: Optional[str] = Field(default=None, description="Time the beam dwells on a given angular position", examples=['50 ms'])
    pulses_per_dwell: Optional[int] = Field(default=None, description="Number of pulses transmitted per beam dwell", examples=[20])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RfEmissionEntity(BaseModel):
    """RF Emission — Observed or catalogued RF emission with power and spectral characteristics
    """
    model_config = ConfigDict(ontology_name="RF_EMISSION", graph_id_fields=[], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name or identifier of the RF emission", examples=['Tombstone Search Mode'])
    emitter_id: Optional[str] = Field(default=None, description="Unique emitter identifier from ELINT database")
    center_frequency_mhz: Optional[float] = Field(default=None, description="Center frequency of the emission in MHz", examples=[9400])
    bandwidth_mhz: Optional[float] = Field(default=None, description="Signal bandwidth in MHz", examples=[50])
    ERP_kw: Optional[float] = Field(default=None, description="Effective radiated power in kilowatts", examples=[1200])
    tx_peak_power_kw: Optional[float] = Field(default=None, description="Transmitter peak power in kilowatts", examples=[150])
    duty_cycle: Optional[float] = Field(default=None, description="Duty cycle ratio (0.0 to 1.0)", examples=[0.05])
    polarization: Optional[str] = Field(default=None, description="Polarization of the transmitted signal", examples=['Linear vertical'])
    radome_loss_db: Optional[float] = Field(default=None, description="Signal loss through the radome in dB", examples=[0.5])
    total_system_losses_db: Optional[float] = Field(default=None, description="Total system losses from transmitter to antenna in dB", examples=[3.2])
    rf_signatures: List["RfSignatureEntity"] = edge(label="HAS_SIGNATURE", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class WaveformEntity(BaseModel):
    """Waveform — Radar or communications waveform with pulse/timing characteristics
    """
    model_config = ConfigDict(ontology_name="WAVEFORM", graph_id_fields=['waveform_name'], identity_scope="document", dodaf_parent="EMEntity", is_entity=True)

    waveform_name: str = Field(..., description="Name of the waveform mode", examples=['Search Mode 1', 'Search Mode 1'])
    waveform_family: Optional[str] = Field(default=None, description="Family or class of the waveform", json_schema_extra={"enum": ["PULSE_DOPPLER", "FMCW", "CW", "LFM_CHIRP", "PHASE_CODED", "BURST", "PULSE"]})
    nominal_pulse_duration_us: Optional[float] = Field(default=None, description="Nominal pulse duration in microseconds", examples=[10.0])
    pulse_duration_limits: Optional[str] = Field(default=None, description="Pulse duration range limits (min-max) in microseconds", examples=['1-100 us'])
    nominal_PRI_us: Optional[float] = Field(default=None, description="Nominal pulse repetition interval in microseconds", examples=[500])
    PRI_limits: Optional[str] = Field(default=None, description="PRI range limits (min-max) in microseconds", examples=['200-1000 us'])
    PRF_limits: Optional[str] = Field(default=None, description="PRF range limits (min-max) in Hz", examples=['1000-5000 Hz'])
    duty_cycle: Optional[float] = Field(default=None, description="Waveform duty cycle ratio (0.0 to 1.0)", examples=[0.02])
    modulations: List["ModulationEntity"] = edge(label="USES_MODULATION", default_factory=list)
    rf_signatures: List["RfSignatureEntity"] = edge(label="HAS_SIGNATURE", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ScanPatternEntity(BaseModel):
    """Scan Pattern — Antenna scan type and timing
    """
    model_config = ConfigDict(ontology_name="SCAN_PATTERN", graph_id_fields=[], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    scan_type: Optional[str] = Field(default=None, description="Type of antenna scan pattern", examples=['Conical scan'])
    scan_period_limits: Optional[str] = Field(default=None, description="Scan period range limits (min-max)", examples=['2-6 s'])
    slew_rate: Optional[str] = Field(default=None, description="Antenna slew rate", examples=['25 deg/s'])
    illumination_time: Optional[str] = Field(default=None, description="Time target is illuminated per scan", examples=['40 ms'])
    dwell_time: Optional[str] = Field(default=None, description="Time the beam dwells at each position", examples=['50 ms'])
    pulses_per_dwell: Optional[int] = Field(default=None, description="Number of pulses per beam dwell", examples=[20])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class AntennaEntity(BaseModel):
    """Antenna — Antenna system with polarization, beamwidth, gain, and geometry
    """
    model_config = ConfigDict(ontology_name="ANTENNA", graph_id_fields=['name'], identity_scope="document", dodaf_parent="EMEntity", is_entity=True)

    name: str = Field(..., description="Name or designation of the antenna", examples=['Main Array Antenna', 'Main Array Antenna'])
    antenna_type: Optional[str] = Field(default=None, description="Physical type of antenna", json_schema_extra={"enum": ["PHASED_ARRAY", "DISH", "DIPOLE", "HORN", "PATCH", "YAGI", "CONFORMAL", "SLOTTED_ARRAY", "ESA"]})
    tx_polarization: Optional[str] = Field(default=None, description="Transmit polarization", examples=['Linear vertical'])
    rx_polarization: Optional[str] = Field(default=None, description="Receive polarization", examples=['Linear vertical'])
    number_of_beams: Optional[int] = Field(default=None, description="Number of simultaneous beams supported", examples=[1])
    beamwidth_az_deg: Optional[float] = Field(default=None, description="Azimuth beamwidth in degrees", examples=[1.5])
    beamwidth_el_deg: Optional[float] = Field(default=None, description="Elevation beamwidth in degrees", examples=[1.5])
    aperture_distribution_az: Optional[str] = Field(default=None, description="Aperture amplitude distribution in azimuth", examples=['Taylor -40 dB'])
    aperture_distribution_el: Optional[str] = Field(default=None, description="Aperture amplitude distribution in elevation", examples=['Taylor -40 dB'])
    gain_dbi: Optional[float] = Field(default=None, description="Peak antenna gain in dBi", examples=[38.0])
    backlobe_level_db: Optional[float] = Field(default=None, description="Backlobe level relative to main beam in dB", examples=[-40])
    dimension_horizontal_m: Optional[float] = Field(default=None, description="Horizontal aperture dimension in meters", examples=[3.0])
    dimension_vertical_m: Optional[float] = Field(default=None, description="Vertical aperture dimension in meters", examples=[3.0])
    height_m: Optional[float] = Field(default=None, description="Antenna height above ground in meters", examples=[8.0])
    min_elevation_deg: Optional[float] = Field(default=None, description="Minimum elevation angle in degrees", examples=[-3])
    max_elevation_deg: Optional[float] = Field(default=None, description="Maximum elevation angle in degrees", examples=[90])
    first_sidelobe_level_az_db: Optional[float] = Field(default=None, description="First azimuth sidelobe level relative to main beam in dB", examples=[-25])
    first_sidelobe_level_el_db: Optional[float] = Field(default=None, description="First elevation sidelobe level relative to main beam in dB", examples=[-25])
    radar_systems: List["RadarSystemEntity"] = edge(label="PART_OF", default_factory=list)
    rf_emissions: List["RfEmissionEntity"] = edge(label="RADIATES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class TransmitterEntity(BaseModel):
    """Transmitter — Radar or communications transmitter with power and loss characteristics
    """
    model_config = ConfigDict(ontology_name="TRANSMITTER", graph_id_fields=['name'], identity_scope="document", dodaf_parent="EMEntity", is_entity=True)

    name: str = Field(..., description="Name or designation of the transmitter", examples=['Main Transmitter Unit', 'Main Transmitter Unit'])
    peak_power_ERP_kw: Optional[float] = Field(default=None, description="Peak effective radiated power in kilowatts", examples=[1200])
    peak_power_at_transmitter_kw: Optional[float] = Field(default=None, description="Peak power at the transmitter output in kilowatts", examples=[150])
    duty_cycle: Optional[float] = Field(default=None, description="Transmitter duty cycle (0.0 to 1.0)", examples=[0.05])
    tx_line_loss_db: Optional[float] = Field(default=None, description="Transmit feed line loss in dB", examples=[1.5])
    other_system_losses_db: Optional[float] = Field(default=None, description="Other system losses (filters, switches) in dB", examples=[0.5])
    total_system_losses_db: Optional[float] = Field(default=None, description="Total system losses from transmitter to antenna in dB", examples=[2.0])
    radar_systems: List["RadarSystemEntity"] = edge(label="PART_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ReceiverEntity(BaseModel):
    """Receiver — Radar or communications receiver with noise and sensitivity characteristics
    """
    model_config = ConfigDict(ontology_name="RECEIVER", graph_id_fields=['name'], identity_scope="document", dodaf_parent="EMEntity", is_entity=True)

    name: str = Field(..., description="Name or designation of the receiver", examples=['Main Receiver Unit', 'Main Receiver Unit'])
    radome_loss_db: Optional[float] = Field(default=None, description="Signal loss through the radome in dB", examples=[0.5])
    clutter_improvement_factor_db: Optional[float] = Field(default=None, description="Improvement factor for clutter rejection in dB", examples=[30])
    noise_figure_db: Optional[float] = Field(default=None, description="Receiver noise figure in dB", examples=[3.5])
    minimum_discernible_signal_dbm: Optional[float] = Field(default=None, description="Minimum detectable signal level in dBm", examples=[-110])
    receive_line_loss_db: Optional[float] = Field(default=None, description="Receive feed line loss in dB", examples=[1.0])
    peak_power_noise_bandwidth_mhz: Optional[float] = Field(default=None, description="Noise bandwidth for peak power measurement in MHz", examples=[2.0])
    average_power_noise_bandwidth_mhz: Optional[float] = Field(default=None, description="Noise bandwidth for average power measurement in MHz", examples=[0.5])
    radar_systems: List["RadarSystemEntity"] = edge(label="PART_OF", default_factory=list)
    rf_emissions: List["RfEmissionEntity"] = edge(label="RECEIVES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class IfAmplifierEntity(BaseModel):
    """IF Amplifier — Intermediate frequency amplifier stage
    """
    model_config = ConfigDict(ontology_name="IF_AMPLIFIER", graph_id_fields=[], identity_scope="global", dodaf_parent="EMEntity", is_entity=True)

    stage_number: Optional[int] = Field(default=None, description="Stage number in the IF amplifier chain", examples=[1])
    bandwidth_3db_mhz: Optional[float] = Field(default=None, description="3 dB bandwidth of the IF stage in MHz", examples=[5.0])
    receivers: List["ReceiverEntity"] = edge(label="PART_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SignalProcessingChainEntity(BaseModel):
    """Signal Processing Chain — Signal processing subsystem with detection, integration, and Doppler parameters
    """
    model_config = ConfigDict(ontology_name="SIGNAL_PROCESSING_CHAIN", graph_id_fields=['name'], identity_scope="document", dodaf_parent="EMEntity", is_entity=True)

    name: str = Field(..., description="Name of the signal processing chain", examples=['Main Processing Chain', 'Main Processing Chain'])
    matched_filter_detection_loss_db: Optional[float] = Field(default=None, description="Loss relative to ideal matched filter detection in dB", examples=[1.5])
    STC: Optional[str] = Field(default=None, description="Sensitivity Time Control configuration", examples=['Range-dependent, 20 dB dynamic range'])
    pulse_compression_ratio: Optional[float] = Field(default=None, description="Ratio of uncompressed to compressed pulse width", examples=[100])
    pulse_compression_gain_db: Optional[float] = Field(default=None, description="Processing gain from pulse compression in dB", examples=[20])
    pulse_compression_weighting_function: Optional[str] = Field(default=None, description="Sidelobe reduction weighting function", examples=['Hamming'])
    coherent_processing_interval: Optional[str] = Field(default=None, description="Duration of coherent integration period", examples=['50 ms'])
    filter_response_type: Optional[str] = Field(default=None, description="Doppler filter bank response type", examples=['FFT with Chebyshev window'])
    doppler_filter_bandwidth_hz: Optional[float] = Field(default=None, description="Individual Doppler filter bandwidth in Hz", examples=[50])
    approaching_doppler_coverage: Optional[str] = Field(default=None, description="Doppler velocity coverage for approaching targets", examples=['50-2000 m/s'])
    pulses_on_target: Optional[int] = Field(default=None, description="Total number of pulses hitting the target per scan", examples=[20])
    predetection_pulses: Optional[int] = Field(default=None, description="Number of pulses coherently integrated before detection", examples=[16])
    predetection_integration_gain_db: Optional[float] = Field(default=None, description="Gain from coherent pre-detection integration in dB", examples=[12])
    postdetection_pulses: Optional[int] = Field(default=None, description="Number of detections non-coherently integrated after detection", examples=[4])
    postdetection_integration_gain_db: Optional[float] = Field(default=None, description="Gain from non-coherent post-detection integration in dB", examples=[4])
    effective_number_of_pulses_integrated: Optional[int] = Field(default=None, description="Effective total pulses integrated (pre + post)", examples=[20])
    minimum_snr_required_db: Optional[float] = Field(default=None, description="Minimum signal-to-noise ratio required for detection in dB", examples=[13])
    MTI_improvement_factor_db: Optional[float] = Field(default=None, description="Moving Target Indication improvement factor in dB", examples=[30])
    drop_track_threshold_improvement_db: Optional[float] = Field(default=None, description="Additional SNR margin for maintaining track in dB", examples=[3])
    radar_systems: List["RadarSystemEntity"] = edge(label="PART_OF", default_factory=list)
    rf_emissions: List["RfEmissionEntity"] = edge(label="PROCESSES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


# ----------------------------------------------------------------------
# Layer 4
# ----------------------------------------------------------------------

class GuidanceMethodEntity(BaseModel):
    """Guidance Method — Missile guidance scheme (command, beam-riding, homing)
    """
    model_config = ConfigDict(ontology_name="GUIDANCE_METHOD", graph_id_fields=['guidance_type'], identity_scope="global", dodaf_parent="WeaponEntity", is_entity=True)

    guidance_type: str = Field(..., description="Type of guidance method", json_schema_extra={"enum": ["COMMAND", "SARH", "ARH", "IR", "BEAM_RIDING", "TVM", "GPS_INS", "DUAL_MODE"]})
    firing_doctrine: Optional[str] = Field(default=None, description="Firing doctrine (shoot-look-shoot, ripple, etc.)", examples=['Shoot-look-shoot'])
    track_quality: Optional[str] = Field(default=None, description="Required track quality for guidance", examples=['Fire-control quality track required'])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SeekerEntity(BaseModel):
    """Seeker — Missile terminal guidance seeker head
    """
    model_config = ConfigDict(ontology_name="SEEKER", graph_id_fields=['seeker_nomenclature'], identity_scope="document", dodaf_parent="WeaponEntity", is_entity=True)

    seeker_nomenclature: str = Field(..., description="Designation or nomenclature of the seeker", examples=['Ka-band active seeker', 'Ka-band active seeker'])
    seeker_ELNOT: Optional[str] = Field(default=None, description="ELNOT identifier for the seeker emitter")
    seeker_DIEQP: Optional[str] = Field(default=None, description="DIEQP code for the seeker")
    seeker_type: Optional[str] = Field(default=None, description="Guidance technology used by the seeker", json_schema_extra={"enum": ["ACTIVE_RADAR", "SEMI_ACTIVE_RADAR", "IR", "DUAL_MODE", "ARM", "GPS_INS", "COMMAND"]})
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class MissilePerformanceEntity(BaseModel):
    """Missile Performance — Missile kinematic and engagement performance envelope
    """
    model_config = ConfigDict(ontology_name="MISSILE_PERFORMANCE", graph_id_fields=[], identity_scope="global", dodaf_parent="WeaponEntity", is_entity=True)

    maximum_range_km: Optional[float] = Field(default=None, description="Maximum engagement range in kilometers", examples=[160])
    minimum_range_km: Optional[float] = Field(default=None, description="Minimum engagement range in kilometers", examples=[3])
    maximum_intercept_range_km: Optional[float] = Field(default=None, description="Maximum intercept range in kilometers", examples=[150])
    maximum_recommended_intercept_range_km: Optional[float] = Field(default=None, description="Maximum recommended intercept range for reliable kill in km", examples=[120])
    maximum_altitude_km: Optional[float] = Field(default=None, description="Maximum engagement altitude in kilometers", examples=[25])
    minimum_altitude_m: Optional[float] = Field(default=None, description="Minimum engagement altitude in meters", examples=[60])
    maximum_launch_angle_deg: Optional[float] = Field(default=None, description="Maximum off-boresight launch angle in degrees", examples=[360])
    intercept_assessment_time_s: Optional[float] = Field(default=None, description="Time to assess intercept success in seconds", examples=[5])
    time_to_go_s: Optional[float] = Field(default=None, description="Estimated time to intercept in seconds", examples=[30])
    acquisition_delay_s: Optional[float] = Field(default=None, description="Delay from cueing to seeker acquisition in seconds", examples=[2])
    handoff_delay_s: Optional[float] = Field(default=None, description="Delay for guidance handoff in seconds", examples=[1])
    track_delay_s: Optional[float] = Field(default=None, description="Delay from acquisition to stable tracking in seconds", examples=[1.5])
    launch_delay_s: Optional[float] = Field(default=None, description="Delay from fire command to missile launch in seconds", examples=[3])
    intra_salvo_time_s: Optional[float] = Field(default=None, description="Time between consecutive missile launches in a salvo in seconds", examples=[1])
    coast_time_s: Optional[float] = Field(default=None, description="Maximum time missile can coast without guidance updates in seconds", examples=[5])
    average_missile_speed_mach: Optional[float] = Field(default=None, description="Average missile speed during flyout in Mach number", examples=[4.1])
    maximum_missile_speed_mach: Optional[float] = Field(default=None, description="Maximum missile speed in Mach number", examples=[5.0])
    maximum_flyout_time_s: Optional[float] = Field(default=None, description="Maximum missile flight time in seconds", examples=[60])
    maximum_offset_deg: Optional[float] = Field(default=None, description="Maximum target offset angle from launcher in degrees", examples=[360])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class MissilePhysicalCharacteristicsEntity(BaseModel):
    """Missile Physical Characteristics — Missile body dimensions and mass
    """
    model_config = ConfigDict(ontology_name="MISSILE_PHYSICAL_CHARACTERISTICS", graph_id_fields=[], identity_scope="global", dodaf_parent="WeaponEntity", is_entity=True)

    body_diameter_m: Optional[float] = Field(default=None, description="Missile body diameter in meters", examples=[0.255])
    overall_length_m: Optional[float] = Field(default=None, description="Overall missile length in meters", examples=[5.2])
    total_mass_kg: Optional[float] = Field(default=None, description="Total missile mass at launch in kilograms", examples=[312])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class PropulsionStackEntity(BaseModel):
    """Propulsion Stack — Complete missile propulsion system (may contain multiple stages)

    KNOWN ANTI-PATTERN: ``PROPULSION_STACK.graph_id_fields = []``. Every instance is a distinct node;
    cannot dedup by logical identity. Flagged for Plan 2 cleanup.
    """
    model_config = ConfigDict(ontology_name="PROPULSION_STACK", graph_id_fields=[], identity_scope="document", dodaf_parent="WeaponEntity", is_entity=True)

    total_burntime_s: Optional[float] = Field(default=None, description="Total burn time across all propulsion stages in seconds", examples=[25])
    propulsion_stages: List["PropulsionStageEntity"] = edge(label="CONTAINS", default_factory=list)
    propulsion_stages: List["PropulsionStageEntity"] = edge(label="HAS_STAGE", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class PropulsionStageEntity(BaseModel):
    """Propulsion Stage — Individual propulsion stage (ejector, booster, sustainer)
    """
    model_config = ConfigDict(ontology_name="PROPULSION_STAGE", graph_id_fields=[], identity_scope="global", dodaf_parent="WeaponEntity", is_entity=True)

    stage_type: Optional[str] = Field(default=None, description="Type of propulsion stage", json_schema_extra={"enum": ["EJECTOR", "BOOSTER", "SUSTAINER"]})
    burn_time_s: Optional[float] = Field(default=None, description="Burn time of the stage in seconds", examples=[12])
    thrust_kn: Optional[float] = Field(default=None, description="Stage thrust in kilonewtons", examples=[180])
    mass_kg: Optional[float] = Field(default=None, description="Stage mass in kilograms", examples=[150])
    diameter_m: Optional[float] = Field(default=None, description="Stage diameter in meters", examples=[0.255])
    length_m: Optional[float] = Field(default=None, description="Stage length in meters", examples=[2.0])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


# ----------------------------------------------------------------------
# Layer 5
# ----------------------------------------------------------------------

class CapabilityEntity(BaseModel):
    """Capability — Operational effect or function achievable by a system (DoDAF DM2)
    """
    model_config = ConfigDict(ontology_name="CAPABILITY", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    capability_name: Optional[str] = Field(default=None, description="Name of the functional capability", examples=['Terminal Phase Guidance'])
    capability_class: Optional[str] = Field(default=None, description="Capability domain or class", json_schema_extra={"enum": ["DETECTION", "TRACKING", "DESIGNATION", "ENGAGEMENT", "FIRE_CONTROL", "AIR_DEFENSE", "ELECTRONIC_ATTACK", "ELECTRONIC_SUPPORT", "SURVEILLANCE", "COMMUNICATIONS"]})
    trl: Optional[int] = Field(default=None, description="Technology Readiness Level on a scale of 1-9", examples=[7])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class RadarPerformanceEntity(BaseModel):
    """Radar Performance — Radar detection, range, and velocity performance envelope
    """
    model_config = ConfigDict(ontology_name="RADAR_PERFORMANCE", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    max_detection_range_1sqm_km: Optional[float] = Field(default=None, description="Maximum detection range against a 1 m^2 RCS target in km", examples=[300])
    min_effective_range_km: Optional[float] = Field(default=None, description="Minimum effective detection range in km", examples=[3])
    max_unambiguous_range_km: Optional[float] = Field(default=None, description="Maximum unambiguous range determined by PRF in km", examples=[400])
    maximum_scope_limit_km: Optional[float] = Field(default=None, description="Maximum display scope range in km", examples=[450])
    maximum_processing_range_km: Optional[float] = Field(default=None, description="Maximum range at which returns are processed in km", examples=[400])
    max_unambiguous_velocity_mps: Optional[float] = Field(default=None, description="Maximum unambiguous radial velocity in m/s", examples=[1500])
    min_range_of_velocity_response_mps: Optional[float] = Field(default=None, description="Minimum radial velocity for detection in m/s", examples=[30])
    max_range_of_velocity_response_mps: Optional[float] = Field(default=None, description="Maximum radial velocity for detection in m/s", examples=[3000])
    minimum_detectable_velocity_mps: Optional[float] = Field(default=None, description="Minimum detectable target velocity due to clutter rejection in m/s", examples=[30])
    maximum_detectable_velocity_mps: Optional[float] = Field(default=None, description="Maximum detectable target velocity (aliasing limit) in m/s", examples=[3000])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class EngagementTimelineEntity(BaseModel):
    """Engagement Timeline — Time sequence from detection through engagement
    """
    model_config = ConfigDict(ontology_name="ENGAGEMENT_TIMELINE", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    detection_to_designate_time_s: Optional[float] = Field(default=None, description="Time from first detection to target designation in seconds", examples=[4])
    designation_to_launch_time_s: Optional[float] = Field(default=None, description="Time from designation to missile launch in seconds", examples=[8])
    acquisition_delay_s: Optional[float] = Field(default=None, description="Delay from cueing to target acquisition in seconds", examples=[2])
    handoff_delay_s: Optional[float] = Field(default=None, description="Delay for handoff between search and track in seconds", examples=[1])
    track_delay_s: Optional[float] = Field(default=None, description="Delay from acquisition to stable track in seconds", examples=[1.5])
    launch_delay_s: Optional[float] = Field(default=None, description="Delay from fire command to missile launch in seconds", examples=[3])
    intercept_assessment_time_s: Optional[float] = Field(default=None, description="Time to assess intercept result in seconds", examples=[5])
    time_to_go_s: Optional[float] = Field(default=None, description="Estimated time from launch to intercept in seconds", examples=[30])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ForceStructureEntity(BaseModel):
    """Force Structure — Military organizational unit or echelon
    """
    model_config = ConfigDict(ontology_name="FORCE_STRUCTURE", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name of the force structure unit", examples=['108th Air Defense Artillery Brigade'])
    echelon: Optional[str] = Field(default=None, description="Military echelon level", examples=['Brigade'])
    service: Optional[str] = Field(default=None, description="Military service branch", examples=['U.S. Army'])
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class AssemblyEntity(BaseModel):
    """Assembly — Grouping of components that function together as a unit
    """
    model_config = ConfigDict(ontology_name="ASSEMBLY", graph_id_fields=[], identity_scope="global", dodaf_parent="MilitaryAsset", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name of the assembly unit", examples=['Antenna Feed Assembly'])
    assembly_number: Optional[str] = Field(default=None, description="Assembly drawing or identification number", examples=['ASM-7891-A'])
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="PART_OF", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class SpecificationEntity(BaseModel):
    """Specification — Measurable performance characteristic
    """
    model_config = ConfigDict(ontology_name="SPECIFICATION", graph_id_fields=['parameter', 'value'], identity_scope="document", dodaf_parent="OperationalEntity", is_entity=True)

    parameter: str = Field(..., description="Name of the measured parameter (e.g. max_range, operating_temperature)", examples=['max_range', 'max_range'])
    value: str = Field(..., description="Numeric value or range of the measurement", examples=['150', '150'])
    unit: Optional[str] = Field(default=None, description="Unit of measurement (SI or military standard)", examples=['km'])
    condition: Optional[str] = Field(default=None, description="Operating conditions under which the spec applies", examples=['sea level, standard atmosphere'])
    source_document: Optional[str] = Field(default=None, description="Document where this specification is defined", examples=['TM 9-1425-386-12'])
    specifications: List["SpecificationEntity"] = edge(label="SUPERSEDES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class StandardEntity(BaseModel):
    """Standard / Specification Document — Reference standard (MIL-STD, MIL-DTL, etc.)
    """
    model_config = ConfigDict(ontology_name="STANDARD", graph_id_fields=[], identity_scope="global", dodaf_parent="DocumentResource", is_entity=True)

    designation: Optional[str] = Field(default=None, description="Official standard designation number (e.g. MIL-STD-1553B)", examples=['MIL-STD-1553B'], pattern='^MIL-[A-Z]+-\\d+[A-Z]?')
    title: Optional[str] = Field(default=None, description="Full title of the standard document", examples=['Digital Time Division Command/Response Multiplex Data Bus'])
    issuing_org: Optional[str] = Field(default=None, description="Organization that published the standard", examples=['Department of Defense'])
    version: Optional[str] = Field(default=None, description="Revision letter or version number", examples=['B'])
    supersedes: Optional[str] = Field(default=None, description="Designation of the standard this one replaces", examples=['MIL-STD-1553A'])
    standards: List["StandardEntity"] = edge(label="SUPERSEDES", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class ProcedureEntity(BaseModel):
    """Procedure — Maintenance, operational, or test procedure
    """
    model_config = ConfigDict(ontology_name="PROCEDURE", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name or title of the procedure", examples=['Radar Antenna Alignment Procedure'])
    type: Optional[str] = Field(default=None, description="Category of procedure", json_schema_extra={"enum": ["MAINTENANCE", "OPERATIONAL", "TEST", "CALIBRATION", "INSPECTION"]})
    periodicity: Optional[str] = Field(default=None, description="How often the procedure must be performed", examples=['Semi-annual'])
    skill_level: Optional[str] = Field(default=None, description="Required maintenance skill level", examples=['20C (Patriot Repairer)'])
    organizations: List["OrganizationEntity"] = edge(label="OPERATED_BY", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class FailureModeEntity(BaseModel):
    """Failure Mode — Known failure mode with FMECA severity
    """
    model_config = ConfigDict(ontology_name="FAILURE_MODE", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Short name of the failure mode", examples=['TWT Power Degradation'])
    description: Optional[str] = Field(default=None, description="Detailed description of how the failure manifests", examples=['Gradual loss of transmit power due to cathode erosion'])
    fmeca_severity: Optional[int] = Field(default=None, description="MIL-STD-1629 severity category (1=catastrophic, 4=minor)", examples=[2])
    detection_method: Optional[str] = Field(default=None, description="How this failure is detected", examples=['BIT fault code 47, power output below threshold'])
    components: List["ComponentEntity"] = edge(label="AFFECTS", default_factory=list)
    subsystems: List["SubsystemEntity"] = edge(label="AFFECTS", default_factory=list)
    equipment_systems: List["EquipmentSystemEntity"] = edge(label="AFFECTS", default_factory=list)
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})

class TestEventEntity(BaseModel):
    """Test Event — Test or evaluation event with outcomes
    """
    model_config = ConfigDict(ontology_name="TEST_EVENT", graph_id_fields=[], identity_scope="global", dodaf_parent="OperationalEntity", is_entity=True)

    name: Optional[str] = Field(default=None, description="Name or designation of the test event", examples=['FET-10 Flight Test'])
    date: Optional[str] = Field(default=None, description="Date the test was conducted (YYYY-MM-DD)", examples=['2024-03-15'])
    location: Optional[str] = Field(default=None, description="Test range or facility location", examples=['White Sands Missile Range, NM'])
    test_type: Optional[str] = Field(default=None, description="Category of test event", json_schema_extra={"enum": ["DT", "OT", "IOT", "LFT", "DEVELOPMENTAL"]})
    outcome: Optional[str] = Field(default=None, description="Overall test result", json_schema_extra={"enum": ["PASS", "FAIL", "PARTIAL", "INCONCLUSIVE"]})
    confidence: Optional[float] = Field(default=None, description="Extraction confidence for this instance, 0–1.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


# ----------------------------------------------------------------------
# ALL_ENTITIES registry
# ----------------------------------------------------------------------

ALL_ENTITIES: dict[str, type[BaseModel]] = {
    "DOCUMENT": DocumentEntity,
    "SECTION": SectionEntity,
    "FIGURE": FigureEntity,
    "TABLE": TableEntity,
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
    "FREQUENCY_BAND": FrequencyBandEntity,
    "MODULATION": ModulationEntity,
    "RF_SIGNATURE": RfSignatureEntity,
    "RF_EMISSION": RfEmissionEntity,
    "WAVEFORM": WaveformEntity,
    "SCAN_PATTERN": ScanPatternEntity,
    "ANTENNA": AntennaEntity,
    "TRANSMITTER": TransmitterEntity,
    "RECEIVER": ReceiverEntity,
    "IF_AMPLIFIER": IfAmplifierEntity,
    "SIGNAL_PROCESSING_CHAIN": SignalProcessingChainEntity,
    "GUIDANCE_METHOD": GuidanceMethodEntity,
    "SEEKER": SeekerEntity,
    "MISSILE_PERFORMANCE": MissilePerformanceEntity,
    "MISSILE_PHYSICAL_CHARACTERISTICS": MissilePhysicalCharacteristicsEntity,
    "PROPULSION_STACK": PropulsionStackEntity,
    "PROPULSION_STAGE": PropulsionStageEntity,
    "CAPABILITY": CapabilityEntity,
    "RADAR_PERFORMANCE": RadarPerformanceEntity,
    "ENGAGEMENT_TIMELINE": EngagementTimelineEntity,
    "FORCE_STRUCTURE": ForceStructureEntity,
    "ASSEMBLY": AssemblyEntity,
    "SPECIFICATION": SpecificationEntity,
    "STANDARD": StandardEntity,
    "PROCEDURE": ProcedureEntity,
    "FAILURE_MODE": FailureModeEntity,
    "TEST_EVENT": TestEventEntity,
}

# Rebuild forward references for all entity classes.
for _cls in ALL_ENTITIES.values():
    _cls.model_rebuild()
