# Relationship Placement Table

**Generated:** 2026-04-15 (Plan v32 Task 9.5).

Every triple in `VALIDATION_MATRIX` (127 unique entries; ontology YAML lists 128 with one duplicate `(RADAR_SYSTEM, INSTALLED_ON, PLATFORM)`) gets a placement decision: typed-edge field on a specific class, post-merge derived edge, or system_links DTO. This is the reference input for `entities.py` Phase 2 (Tasks 14–18).

**Ontology YAML duplicate flagged:** `validation_matrix` in `ontology_bundles/air_defense_v3/ontology.yaml` contains `(RADAR_SYSTEM, INSTALLED_ON, PLATFORM)` twice. When Phase 2 Task 12 defines `VALIDATION_MATRIX` as a `frozenset`, the duplicate is naturally collapsed to 127 unique triples. No action required in Phase 1.5 — this table dedupes the row for clarity.

## Placement kinds

- **typed_edge:** a Pydantic `edge(label=...)` field on the named owning class. Extracted during the owning class's pass. Default for most triples.
- **post_merge:** not a Pydantic field. Created by `derive_rules` / `derive_structure_links` after merge. Examples: `HAS_PROVENANCE` (global-entity ↔ DOCUMENT), `MENTIONED_IN` (entity ↔ TextChunk), `NEXT_CHUNK` (chunk ↔ chunk).
- **system_links:** DTO exception — represented via `SystemLinkRelationship(from_ref_id, to_ref_id, rel_type)` in the system_links pass. Retained per Decision 4. Examples: `ASSOCIATED_WITH`, `CUES`.

## Placement rules (applied)

1. Direction follows the edge name — "HAS_X" places the edge field on the source class.
2. No inverse fields (back-traversal is a query-time concern, not a schema concern).
3. Cardinality heuristic: `HAS_PROPULSION_STACK`, `SPECIFIED_BY`, and generic `HAS_*` default to `List[...]`. Singletons: `HAS_GUIDANCE`, `HAS_WARHEAD`, `HAS_SEEKER`, `HAS_FIRE_CONTROL`, `HAS_COMMAND_POST`, `HAS_LAUNCHER`, `INSTALLED_ON`, `MOUNTED_ON`.
4. Max nesting depth: 3 (per docs). Current placement stays at depth ≤ 2 for all entries — no flattening required.

## Table

| kind | source | relationship | target | owning class | field | cardinality | nullable | rationale |
|---|---|---|---|---|---|---|---|---|
| typed_edge | RADAR_SYSTEM | INSTALLED_ON | PLATFORM | RadarSystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| system_links | RADAR_SYSTEM | ASSOCIATED_WITH | MISSILE_SYSTEM |  |  |  |  |  |
| system_links | RADAR_SYSTEM | ASSOCIATED_WITH | AIR_DEFENSE_ARTILLERY_SYSTEM |  |  |  |  |  |
| typed_edge | RADAR_SYSTEM | USES_WAVEFORM | WAVEFORM | RadarSystemEntity | waveforms | List[WaveformEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | EMITS | RF_EMISSION | RadarSystemEntity | rf_emissions | List[RfEmissionEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_ANTENNA | ANTENNA | RadarSystemEntity | antennas | List[AntennaEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_RECEIVER | RECEIVER | RadarSystemEntity | receivers | List[ReceiverEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_TRANSMITTER | TRANSMITTER | RadarSystemEntity | transmitters | List[TransmitterEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_SCAN | SCAN_PATTERN | RadarSystemEntity | scan_patterns | List[ScanPatternEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_PROCESSING_CHAIN | SIGNAL_PROCESSING_CHAIN | RadarSystemEntity | signal_processing_chains | List[SignalProcessingChainEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_PERFORMANCE | RADAR_PERFORMANCE | RadarSystemEntity | radar_performances | List[RadarPerformanceEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | OPERATES_IN_BAND | FREQUENCY_BAND | RadarSystemEntity | frequency_bands | List[FrequencyBandEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | PROVIDES | CAPABILITY | RadarSystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| system_links | RADAR_SYSTEM | CUES | MISSILE_SYSTEM |  |  |  |  |  |
| typed_edge | RADAR_SYSTEM | HAS_SIGNATURE | RF_SIGNATURE | RadarSystemEntity | rf_signatures | List[RfSignatureEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | HAS_TIMELINE | ENGAGEMENT_TIMELINE | RadarSystemEntity | engagement_timelines | List[EngagementTimelineEntity] | yes (default_factory=list) |  |
| system_links | MISSILE_SYSTEM | ASSOCIATED_WITH | RADAR_SYSTEM |  |  |  |  |  |
| typed_edge | MISSILE_SYSTEM | HAS_GUIDANCE | GUIDANCE_METHOD | MissileSystemEntity | guidance_method | Optional[GuidanceMethodEntity] | yes (default=None) |  |
| typed_edge | MISSILE_SYSTEM | HAS_SEEKER | SEEKER | MissileSystemEntity | seeker | Optional[SeekerEntity] | yes (default=None) |  |
| typed_edge | MISSILE_SYSTEM | HAS_PROPULSION | PROPULSION_STACK | MissileSystemEntity | propulsion_stacks | List[PropulsionStackEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | HAS_PERFORMANCE | MISSILE_PERFORMANCE | MissileSystemEntity | missile_performances | List[MissilePerformanceEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | INSTALLED_ON | PLATFORM | MissileSystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| typed_edge | MISSILE_SYSTEM | PROVIDES | CAPABILITY | MissileSystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | DEFENDS | PLATFORM | MissileSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| system_links | AIR_DEFENSE_ARTILLERY_SYSTEM | ASSOCIATED_WITH | RADAR_SYSTEM |  |  |  |  |  |
| typed_edge | AIR_DEFENSE_ARTILLERY_SYSTEM | INSTALLED_ON | PLATFORM | AirDefenseArtillerySystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| typed_edge | AIR_DEFENSE_ARTILLERY_SYSTEM | PROVIDES | CAPABILITY | AirDefenseArtillerySystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | CONTAINS | RADAR_SYSTEM | IntegratedAirDefenseSystemEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | CONTAINS | MISSILE_SYSTEM | IntegratedAirDefenseSystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | CONTAINS | AIR_DEFENSE_ARTILLERY_SYSTEM | IntegratedAirDefenseSystemEntity | air_defense_artillery_systems | List[AirDefenseArtillerySystemEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | DEPLOYED_ON | PLATFORM | IntegratedAirDefenseSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | PROVIDES | CAPABILITY | IntegratedAirDefenseSystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| typed_edge | WAVEFORM | USES_MODULATION | MODULATION | WaveformEntity | modulations | List[ModulationEntity] | yes (default_factory=list) |  |
| typed_edge | WAVEFORM | HAS_SIGNATURE | RF_SIGNATURE | WaveformEntity | rf_signatures | List[RfSignatureEntity] | yes (default_factory=list) |  |
| typed_edge | TRANSMITTER | PART_OF | RADAR_SYSTEM | TransmitterEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ANTENNA | PART_OF | RADAR_SYSTEM | AntennaEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ANTENNA | RADIATES | RF_EMISSION | AntennaEntity | rf_emissions | List[RfEmissionEntity] | yes (default_factory=list) |  |
| typed_edge | RECEIVER | PART_OF | RADAR_SYSTEM | ReceiverEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | RECEIVER | RECEIVES | RF_EMISSION | ReceiverEntity | rf_emissions | List[RfEmissionEntity] | yes (default_factory=list) |  |
| typed_edge | SIGNAL_PROCESSING_CHAIN | PART_OF | RADAR_SYSTEM | SignalProcessingChainEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | SIGNAL_PROCESSING_CHAIN | PROCESSES | RF_EMISSION | SignalProcessingChainEntity | rf_emissions | List[RfEmissionEntity] | yes (default_factory=list) |  |
| typed_edge | IF_AMPLIFIER | PART_OF | RECEIVER | IfAmplifierEntity | receivers | List[ReceiverEntity] | yes (default_factory=list) |  |
| typed_edge | RF_EMISSION | HAS_SIGNATURE | RF_SIGNATURE | RfEmissionEntity | rf_signatures | List[RfSignatureEntity] | yes (default_factory=list) |  |
| typed_edge | PROPULSION_STACK | CONTAINS | PROPULSION_STAGE | PropulsionStackEntity | propulsion_stages | List[PropulsionStageEntity] | yes (default_factory=list) |  |
| typed_edge | PROPULSION_STACK | HAS_STAGE | PROPULSION_STAGE | PropulsionStackEntity | propulsion_stages | List[PropulsionStageEntity] | yes (default_factory=list) |  |
| typed_edge | LAUNCHER_SYSTEM | LAUNCHES | MISSILE_SYSTEM | LauncherSystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | LAUNCHER_SYSTEM | INSTALLED_ON | PLATFORM | LauncherSystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| system_links | FIRE_CONTROL_SYSTEM | CUES | MISSILE_SYSTEM |  |  |  |  |  |
| typed_edge | FIRE_CONTROL_SYSTEM | GUIDES | MISSILE_SYSTEM | FireControlSystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | FIRE_CONTROL_SYSTEM | INSTALLED_ON | PLATFORM | FireControlSystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| typed_edge | ELECTRONIC_WARFARE_SYSTEM | INSTALLED_ON | PLATFORM | ElectronicWarfareSystemEntity | platform | Optional[PlatformEntity] | yes (default=None) |  |
| typed_edge | ELECTRONIC_WARFARE_SYSTEM | OPERATES_IN_BAND | FREQUENCY_BAND | ElectronicWarfareSystemEntity | frequency_bands | List[FrequencyBandEntity] | yes (default_factory=list) |  |
| typed_edge | ELECTRONIC_WARFARE_SYSTEM | PROVIDES | CAPABILITY | ElectronicWarfareSystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| typed_edge | PLATFORM | OPERATED_BY | ORGANIZATION | PlatformEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | SUPPORTED_BY | DOCUMENT | AssertionEntity | documents | List[DocumentEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | SUPPORTED_BY | SECTION | AssertionEntity | sections | List[SectionEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | SUPPORTED_BY | FIGURE | AssertionEntity | figures | List[FigureEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | SUPPORTED_BY | TABLE | AssertionEntity | tables | List[TableEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | SUPPORTED_BY | SPREADSHEET | AssertionEntity | spreadsheets | List[SpreadsheetEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | ABOUT | EQUIPMENT_SYSTEM | AssertionEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | ABOUT | RADAR_SYSTEM | AssertionEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | ABOUT | MISSILE_SYSTEM | AssertionEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | ABOUT | PLATFORM | AssertionEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | ABOUT | COMPONENT | AssertionEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| post_merge | EQUIPMENT_SYSTEM | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| post_merge | RADAR_SYSTEM | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| post_merge | MISSILE_SYSTEM | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| post_merge | PLATFORM | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| post_merge | COMPONENT | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| post_merge | SUBSYSTEM | MENTIONED_IN | DOCUMENT |  |  |  |  |  |
| typed_edge | SUBSYSTEM | PART_OF | EQUIPMENT_SYSTEM | SubsystemEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | COMPONENT | PART_OF | SUBSYSTEM | ComponentEntity | subsystems | List[SubsystemEntity] | yes (default_factory=list) |  |
| typed_edge | ASSEMBLY | PART_OF | EQUIPMENT_SYSTEM | AssemblyEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | WEAPON_SYSTEM | CONTAINS | SUBSYSTEM | WeaponSystemEntity | subsystems | List[SubsystemEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | CONTAINS | COMPONENT | EquipmentSystemEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | HAS_SUBSYSTEM | SUBSYSTEM | EquipmentSystemEntity | subsystems | List[SubsystemEntity] | yes (default_factory=list) |  |
| typed_edge | WEAPON_SYSTEM | HAS_SUBSYSTEM | SUBSYSTEM | WeaponSystemEntity | subsystems | List[SubsystemEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | HAS_COMPONENT | COMPONENT | EquipmentSystemEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| typed_edge | WEAPON_SYSTEM | HAS_COMPONENT | COMPONENT | WeaponSystemEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| typed_edge | SUBSYSTEM | HAS_COMPONENT | COMPONENT | SubsystemEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | PROVIDES | CAPABILITY | EquipmentSystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| system_links | RADAR_SYSTEM | ASSOCIATED_WITH | ELECTRONIC_WARFARE_SYSTEM |  |  |  |  |  |
| typed_edge | EQUIPMENT_SYSTEM | MANUFACTURED_BY | ORGANIZATION | EquipmentSystemEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | COMPONENT | MANUFACTURED_BY | ORGANIZATION | ComponentEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | PLATFORM | MANUFACTURED_BY | ORGANIZATION | PlatformEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | DERIVED_FROM | DOCUMENT | AssertionEntity | documents | List[DocumentEntity] | yes (default_factory=list) |  |
| typed_edge | DOCUMENT | DERIVED_FROM | DOCUMENT | DocumentEntity | documents | List[DocumentEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | IS_A | EQUIPMENT_SYSTEM | RadarSystemEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | IS_A | WEAPON_SYSTEM | MissileSystemEntity | weapon_systems | List[WeaponSystemEntity] | yes (default_factory=list) |  |
| typed_edge | AIR_DEFENSE_ARTILLERY_SYSTEM | IS_A | WEAPON_SYSTEM | AirDefenseArtillerySystemEntity | weapon_systems | List[WeaponSystemEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | INSTANCE_OF | EQUIPMENT_SYSTEM | EquipmentSystemEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | PLATFORM | INSTANCE_OF | PLATFORM | PlatformEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | ALIAS_OF | RADAR_SYSTEM | RadarSystemEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | ALIAS_OF | MISSILE_SYSTEM | MissileSystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | ALIAS_OF | EQUIPMENT_SYSTEM | EquipmentSystemEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | DOCUMENT | SUPERSEDES | DOCUMENT | DocumentEntity | documents | List[DocumentEntity] | yes (default_factory=list) |  |
| typed_edge | STANDARD | SUPERSEDES | STANDARD | StandardEntity | standards | List[StandardEntity] | yes (default_factory=list) |  |
| typed_edge | SPECIFICATION | SUPERSEDES | SPECIFICATION | SpecificationEntity | specifications | List[SpecificationEntity] | yes (default_factory=list) |  |
| typed_edge | SUBSYSTEM | PART_OF | RADAR_SYSTEM | SubsystemEntity | radar_systems | List[RadarSystemEntity] | yes (default_factory=list) |  |
| typed_edge | SUBSYSTEM | PART_OF | MISSILE_SYSTEM | SubsystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | SUBSYSTEM | PROVIDES | CAPABILITY | SubsystemEntity | capabilities | List[CapabilityEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | SPECIFIED_BY | STANDARD | EquipmentSystemEntity | standards | List[StandardEntity] | yes (default_factory=list) |  |
| typed_edge | COMPONENT | SPECIFIED_BY | STANDARD | ComponentEntity | standards | List[StandardEntity] | yes (default_factory=list) |  |
| typed_edge | PROCEDURE | OPERATED_BY | ORGANIZATION | ProcedureEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | SPECIFIED_BY | SPECIFICATION | EquipmentSystemEntity | specifications | List[SpecificationEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | SPECIFIED_BY | SPECIFICATION | RadarSystemEntity | specifications | List[SpecificationEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | SPECIFIED_BY | SPECIFICATION | MissileSystemEntity | specifications | List[SpecificationEntity] | yes (default_factory=list) |  |
| typed_edge | FAILURE_MODE | AFFECTS | COMPONENT | FailureModeEntity | components | List[ComponentEntity] | yes (default_factory=list) |  |
| typed_edge | FAILURE_MODE | AFFECTS | SUBSYSTEM | FailureModeEntity | subsystems | List[SubsystemEntity] | yes (default_factory=list) |  |
| typed_edge | FAILURE_MODE | AFFECTS | EQUIPMENT_SYSTEM | FailureModeEntity | equipment_systems | List[EquipmentSystemEntity] | yes (default_factory=list) |  |
| typed_edge | EQUIPMENT_SYSTEM | TESTED_IN | TEST_EVENT | EquipmentSystemEntity | test_events | List[TestEventEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | TESTED_IN | TEST_EVENT | RadarSystemEntity | test_events | List[TestEventEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | TESTED_IN | TEST_EVENT | MissileSystemEntity | test_events | List[TestEventEntity] | yes (default_factory=list) |  |
| typed_edge | COMPONENT | TESTED_IN | TEST_EVENT | ComponentEntity | test_events | List[TestEventEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | TRACKS | PLATFORM | RadarSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | FIRE_CONTROL_SYSTEM | TRACKS | PLATFORM | FireControlSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | WEAPON_SYSTEM | ENGAGES | PLATFORM | WeaponSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | MISSILE_SYSTEM | ENGAGES | PLATFORM | MissileSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | AIR_DEFENSE_ARTILLERY_SYSTEM | ENGAGES | PLATFORM | AirDefenseArtillerySystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | DETECTS | PLATFORM | RadarSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | ELECTRONIC_WARFARE_SYSTEM | DETECTS | RF_EMISSION | ElectronicWarfareSystemEntity | rf_emissions | List[RfEmissionEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | DESIGNATES | PLATFORM | RadarSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | FIRE_CONTROL_SYSTEM | DESIGNATES | PLATFORM | FireControlSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | INTEGRATED_AIR_DEFENSE_SYSTEM | SUPPORTS_ENGAGEMENT_OF | PLATFORM | IntegratedAirDefenseSystemEntity | platforms | List[PlatformEntity] | yes (default_factory=list) |  |
| typed_edge | RADAR_SYSTEM | SUPPORTS_ENGAGEMENT_OF | MISSILE_SYSTEM | RadarSystemEntity | missile_systems | List[MissileSystemEntity] | yes (default_factory=list) |  |
| typed_edge | ASSERTION | REVIEWED_BY | ORGANIZATION | AssertionEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |
| typed_edge | DOCUMENT | REVIEWED_BY | ORGANIZATION | DocumentEntity | organizations | List[OrganizationEntity] | yes (default_factory=list) |  |

## Summary

- Total triples: 128
- typed_edge: 115
- post_merge: 6
- system_links: 7

## Usage

Tasks 14–18 declare `entities.py` classes following this table. Deviation during implementation must update this table in the same commit.