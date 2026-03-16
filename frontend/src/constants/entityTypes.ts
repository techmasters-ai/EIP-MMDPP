/** Entity type categories for graph visualization styling. */

export const MILITARY_TYPES = [
  "RadarSystem", "MissileSystem", "AirDefenseArtillerySystem",
  "IntegratedAirDefenseSystem", "ElectronicWarfareSystem", "FireControlSystem",
  "LauncherSystem", "WeaponSystem", "Platform", "Subsystem", "Component",
] as const;

export const EMRF_TYPES = [
  "FrequencyBand", "Waveform", "Modulation", "RFEmission", "RFSignature",
  "Antenna", "Transmitter", "Receiver", "SignalProcessingChain", "ScanPattern",
] as const;

export const WEAPON_TYPES = [
  "Seeker", "GuidanceMethod", "MissilePerformance", "PropulsionStack",
] as const;

export const OPERATIONAL_TYPES = [
  "Capability", "EngagementTimeline", "RadarPerformance",
] as const;

export const REFERENCE_TYPES = [
  "Organization", "Document", "Assertion",
] as const;

/** Map entity type to category string for CSS class. */
export const ENTITY_CATEGORY: Record<string, string> = {};
MILITARY_TYPES.forEach((t) => (ENTITY_CATEGORY[t] = "military"));
EMRF_TYPES.forEach((t) => (ENTITY_CATEGORY[t] = "emrf"));
WEAPON_TYPES.forEach((t) => (ENTITY_CATEGORY[t] = "weapon"));
OPERATIONAL_TYPES.forEach((t) => (ENTITY_CATEGORY[t] = "operational"));
REFERENCE_TYPES.forEach((t) => (ENTITY_CATEGORY[t] = "reference"));

export function getEntityCategory(entityType: string): string {
  return ENTITY_CATEGORY[entityType] || "reference";
}
