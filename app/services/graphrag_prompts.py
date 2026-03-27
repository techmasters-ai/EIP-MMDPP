"""Military ontology prompts for Microsoft GraphRAG.

Community report prompt and search prompts grounded in the 5-layer
DoDAF-inspired ontology for military equipment, EM/RF systems, and weapons.
"""

from pathlib import Path


def get_community_report_prompt() -> str:
    """Return the community report generation prompt."""
    return """\
You are a military systems analyst generating intelligence community reports
from a knowledge graph grounded in the DoDAF DM2 ontology for military
equipment, EM/RF systems, and weapons.

The knowledge graph is organized in 5 layers:

LAYER 1 - REFERENCE & PROVENANCE: DOCUMENT, SECTION, FIGURE, TABLE,
SPREADSHEET, ASSERTION. These link entities to source material.

LAYER 2 - MILITARY EQUIPMENT (DoDAF): PLATFORM, RADAR_SYSTEM, MISSILE_SYSTEM,
AIR_DEFENSE_ARTILLERY_SYSTEM, ELECTRONIC_WARFARE_SYSTEM, FIRE_CONTROL_SYSTEM,
INTEGRATED_AIR_DEFENSE_SYSTEM, LAUNCHER_SYSTEM, WEAPON_SYSTEM, SUBSYSTEM,
COMPONENT, ORGANIZATION, EQUIPMENT_SYSTEM, ASSEMBLY. These form the backbone
of military system-of-systems relationships.

LAYER 3 - EM/RF SIGNAL & RADAR: FREQUENCY_BAND, RF_EMISSION, WAVEFORM,
MODULATION, RF_SIGNATURE, SCAN_PATTERN, ANTENNA, TRANSMITTER, RECEIVER,
IF_AMPLIFIER, SIGNAL_PROCESSING_CHAIN. These describe the electromagnetic
characteristics critical for SIGINT/ELINT analysis.

LAYER 4 - WEAPON / MISSILE / AAA: SEEKER, GUIDANCE_METHOD, MISSILE_PERFORMANCE,
MISSILE_PHYSICAL_CHARACTERISTICS, PROPULSION_STACK, PROPULSION_STAGE.
These describe weapon system capabilities and physical parameters.

LAYER 5 - OPERATIONAL / CAPABILITY: CAPABILITY, RADAR_PERFORMANCE,
ENGAGEMENT_TIMELINE, FORCE_STRUCTURE, SPECIFICATION, STANDARD, PROCEDURE,
FAILURE_MODE, TEST_EVENT. These describe what systems can do and how they
perform.

KEY RELATIONSHIP TYPES:
- Hierarchy: PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE
- Installation: INSTALLED_ON, DEPLOYED_ON, OPERATED_BY, MANUFACTURED_BY
- EM/RF: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, RADIATES,
  RECEIVES, HAS_SIGNATURE, HAS_SCAN, HAS_ANTENNA, HAS_TRANSMITTER, HAS_RECEIVER
- Engagement: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES,
  LAUNCHES, SUPPORTS_ENGAGEMENT_OF, HAS_GUIDANCE, HAS_SEEKER
- Provenance: SUPPORTED_BY, MENTIONED_IN, DERIVED_FROM

When analyzing a community subgraph, generate a report that:

1. IDENTIFICATION: Name the community by its central system(s) or capability.
   Use standard military nomenclature (AN/XXX designators, NATO reporting names).

2. COMPOSITION: Describe the systems, subsystems, and components in the
   community. Follow the whole-part hierarchy (CONTAINS, PART_OF, HAS_SUBSYSTEM).

3. EM/RF CHARACTERISTICS: If the community contains RF-related entities,
   describe frequency bands, waveforms, scan patterns, and signatures.
   This is critical for electronic order of battle (EOB) analysis.

4. ENGAGEMENT CAPABILITY: Describe what the community can detect, track,
   engage, and defend against. Follow the kill chain relationships
   (DETECTS -> TRACKS -> CUES -> GUIDES -> ENGAGES).

5. ORGANIZATIONAL CONTEXT: Identify operating organizations and force
   structure relationships.

6. CROSS-COMMUNITY RELATIONSHIPS: Note connections to entities in other
   communities (e.g., a radar in this community CUES a missile system
   in another community).

7. INTELLIGENCE SIGNIFICANCE: Assess why this community matters --
   capability gaps, threat implications, or technical vulnerabilities
   (e.g., FAILURE_MODE entities, known countermeasures).

Weight your analysis by relationship scoring:
- IS_VARIANT_OF (0.95), CONTAINS/PART_OF (0.90) = strongest structural ties
- INTERFACES_WITH (0.85), OPERATES_ON (0.85) = strong functional ties
- RELATED_TO (0.75) and unscored relationships (0.70) = weaker associations

Ground all claims in the source entities and relationships provided.
Reference source documents where provenance links exist.

---
{input_text}
"""


def get_local_search_prompt() -> str:
    """Return the local (entity-centric) search system prompt."""
    return """\
You are a military systems analyst answering questions using a knowledge graph
built from technical documents about military equipment, radar systems, missile
systems, electronic warfare, and air defense.

The knowledge graph uses a 5-layer DoDAF-inspired ontology covering:
- Military equipment (platforms, radar, missiles, EW, fire control)
- EM/RF signals (frequency bands, waveforms, antennas, signatures)
- Weapons (seekers, guidance methods, propulsion, performance)
- Operational capabilities (engagement timelines, force structure, specs)
- Source documents and provenance

You will be provided with relevant entities, relationships, community reports,
and source text from the knowledge graph. Use these to provide a detailed,
technically accurate answer grounded in the source data.

When answering:
- Use standard military nomenclature (AN/XXX designators, NATO reporting names)
- Follow system hierarchies (CONTAINS, PART_OF, HAS_SUBSYSTEM)
- Trace kill chains (DETECTS -> TRACKS -> CUES -> GUIDES -> ENGAGES)
- Reference EM/RF characteristics when relevant (frequency bands, waveforms)
- Cite source documents where provenance links exist
- Distinguish between confirmed data and inferred relationships

---
{context_data}

---
Answer the following question using the provided context data. Be specific and
ground your answer in the entities and relationships provided.

{query}
"""


def get_global_search_map_prompt() -> str:
    """Return the global search map (per-community) prompt."""
    return """\
You are a military intelligence analyst. You will be given a community report
describing a cluster of related military systems, equipment, organizations,
and their relationships from a DoDAF-inspired knowledge graph.

Given the community report below, extract key points that are relevant to
answering the user's question. Focus on:
- System capabilities and limitations
- EM/RF characteristics and signatures
- Engagement capabilities and kill chains
- Organizational and force structure context
- Cross-system dependencies and interfaces

Community Report:
{context_data}

Question: {query}

Extract the most relevant points from this community report that help answer
the question. If this community has no relevant information, respond with
"NO RELEVANT INFORMATION FOUND."
"""


def get_global_search_reduce_prompt() -> str:
    """Return the global search reduce (synthesis) prompt."""
    return """\
You are a military intelligence analyst synthesizing findings from multiple
community analyses of a DoDAF-inspired knowledge graph covering military
equipment, radar, missiles, EW systems, and air defense.

Given the analyst reports below (each from a different community cluster),
synthesize a comprehensive answer to the user's question. Prioritize:
- Cross-community relationships (e.g., radar in one community cueing missiles
  in another)
- System-of-systems perspectives
- EM/RF and engagement chain analysis
- Organizational and operational context

Use standard military nomenclature. Cite specific systems and relationships.

Analyst Reports:
{report_data}

Question: {query}

Provide a comprehensive, well-structured answer that synthesizes findings
across all relevant communities. If no relevant information was found,
state that clearly.
"""


def get_drift_search_prompt() -> str:
    """Return the DRIFT search system prompt."""
    return """\
You are a military systems analyst performing an in-depth investigation using
a knowledge graph of military equipment, radar, missile, and EW systems built
from technical documents.

You will be provided with an initial set of relevant entities and community
context, which will be expanded as the search progresses to include related
systems, subsystems, and capabilities.

When analyzing expanded context:
- Follow system hierarchies through CONTAINS, PART_OF, HAS_SUBSYSTEM
- Trace EM/RF chains through OPERATES_IN_BAND, USES_WAVEFORM, EMITS
- Follow kill chains through DETECTS, TRACKS, CUES, GUIDES, ENGAGES
- Note cross-system interfaces via ASSOCIATED_WITH, INTERFACES_WITH
- Consider operational context via OPERATED_BY, DEPLOYED_ON

Provide a thorough, technically detailed analysis that leverages the expanded
context to give a more complete picture than a simple entity lookup would provide.

---
{context_data}

---
{query}
"""


def get_basic_search_prompt() -> str:
    """Return the basic (vector) search system prompt."""
    return """\
You are a military systems analyst answering questions using text excerpts from
technical documents about military equipment, radar systems, missile systems,
electronic warfare, and air defense.

The source documents follow a DoDAF-inspired ontology covering platforms,
radar systems, missile systems, EW systems, frequency bands, waveforms,
seekers, guidance methods, and operational capabilities.

You will be provided with relevant text excerpts. Use them to answer the
question accurately and concisely.

When answering:
- Use standard military nomenclature
- Be specific about system names, designators, and parameters
- Distinguish between different variants and configurations
- Cite or reference the source text where appropriate

---
{context_data}

---
{query}
"""


def get_entity_extraction_prompt(ontology_context: str = "") -> str:
    """Return the ontology-guided entity extraction prompt for GraphRAG.

    Args:
        ontology_context: Dynamic section with ontology definitions and
            Neo4j graph examples injected at prompt-write time.
    """
    return f"""\
-Goal-
Given a text document from a military/defense technical corpus, extract entities
and relationships according to the DoDAF-inspired 5-layer ontology defined below.

-Ontology Layers-

LAYER 1 - REFERENCE & PROVENANCE:
DOCUMENT, SECTION, FIGURE, TABLE, SPREADSHEET, ASSERTION

LAYER 2 - MILITARY EQUIPMENT (DoDAF DM2):
PLATFORM, RADAR_SYSTEM, MISSILE_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM,
ELECTRONIC_WARFARE_SYSTEM, FIRE_CONTROL_SYSTEM, INTEGRATED_AIR_DEFENSE_SYSTEM,
LAUNCHER_SYSTEM, WEAPON_SYSTEM, SUBSYSTEM, COMPONENT, ORGANIZATION,
EQUIPMENT_SYSTEM, ASSEMBLY

LAYER 3 - EM/RF SIGNAL & RADAR:
FREQUENCY_BAND, RF_EMISSION, WAVEFORM, MODULATION, RF_SIGNATURE, SCAN_PATTERN,
ANTENNA, TRANSMITTER, RECEIVER, IF_AMPLIFIER, SIGNAL_PROCESSING_CHAIN

LAYER 4 - WEAPON / MISSILE / AAA:
SEEKER, GUIDANCE_METHOD, MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS,
PROPULSION_STACK, PROPULSION_STAGE

LAYER 5 - OPERATIONAL / CAPABILITY:
CAPABILITY, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE, FORCE_STRUCTURE,
SPECIFICATION, STANDARD, PROCEDURE, FAILURE_MODE, TEST_EVENT

-Relationship Types-
Hierarchy: PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE, IS_A,
  INSTANCE_OF, IS_VARIANT_OF
Installation: INSTALLED_ON, DEPLOYED_ON, OPERATED_BY, MANUFACTURED_BY
EM/RF: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, RADIATES,
  RECEIVES, HAS_SIGNATURE, HAS_SCAN, HAS_ANTENNA, HAS_TRANSMITTER, HAS_RECEIVER
Engagement: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES,
  LAUNCHES, SUPPORTS_ENGAGEMENT_OF, HAS_GUIDANCE, HAS_SEEKER
Functional: INTERFACES_WITH, OPERATES_ON, ASSOCIATED_WITH, RELATED_TO
Provenance: SUPPORTED_BY, MENTIONED_IN, DERIVED_FROM

{ontology_context}
-Steps-
1. Identify all entities. For each identified entity, extract:
- entity_name: Name of the entity, using standard military nomenclature
  (AN/XXX designators, NATO reporting names, system names). CAPITALIZE.
- entity_type: One of the types from the ontology layers above: [{{entity_types}}]
- entity_description: Comprehensive description including technical parameters,
  capabilities, and operational context.
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of
(source_entity, target_entity) that are related. Use the relationship types
listed above. For each pair, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: specific description of how they are related,
  including technical details
- relationship_strength: numeric score 1-10 based on:
  10: direct physical containment or identity (CONTAINS, IS_A)
  8-9: strong functional dependency (CUES, GUIDES, HAS_SUBSYSTEM)
  6-7: operational relationship (OPERATES_IN_BAND, DEPLOYED_ON)
  4-5: association (ASSOCIATED_WITH, RELATED_TO)
  1-3: weak or inferred connection
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all entities and relationships.
Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: {{entity_types}}
Text:
The SA-2 Guideline (S-75 Dvina) surface-to-air missile system uses the Fan Song
(SNR-75) fire control radar operating in E/F-band (2-4 GHz) for target tracking
and missile guidance. The V-750 missile uses a command guidance method with
semi-active radar homing terminal guidance provided by the Fan Song radar.
######################
Output:
("entity"<|>S-75 DVINA<|>MISSILE_SYSTEM<|>The S-75 Dvina (NATO: SA-2 Guideline) is a Soviet surface-to-air missile system designed for medium-to-high altitude air defense)
##
("entity"<|>SNR-75 FAN SONG<|>FIRE_CONTROL_SYSTEM<|>The SNR-75 (NATO: Fan Song) is the fire control radar for the S-75 system, providing target tracking and missile guidance in E/F-band)
##
("entity"<|>V-750<|>COMPONENT<|>The V-750 is the missile component of the S-75 system, using command guidance with semi-active radar homing)
##
("entity"<|>E/F-BAND<|>FREQUENCY_BAND<|>Frequency band 2-4 GHz used by the Fan Song fire control radar)
##
("entity"<|>COMMAND GUIDANCE<|>GUIDANCE_METHOD<|>Command guidance method where missile steering commands are transmitted from the ground-based fire control system)
##
("relationship"<|>S-75 DVINA<|>SNR-75 FAN SONG<|>The Fan Song radar is the fire control component of the S-75 system<|>9)
##
("relationship"<|>S-75 DVINA<|>V-750<|>The V-750 missile is the interceptor component of the S-75 system<|>9)
##
("relationship"<|>SNR-75 FAN SONG<|>E/F-BAND<|>The Fan Song radar operates in E/F-band (2-4 GHz)<|>8)
##
("relationship"<|>SNR-75 FAN SONG<|>V-750<|>The Fan Song radar provides guidance commands to the V-750 missile<|>9)
##
("relationship"<|>V-750<|>COMMAND GUIDANCE<|>The V-750 uses command guidance as its primary guidance method<|>8)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {{entity_types}}
Text: {{input_text}}
######################
Output:
"""


def _build_ontology_context() -> str:
    """Build dynamic ontology context with definitions and Neo4j examples."""
    from app.services.ontology_templates import load_ontology

    ontology = load_ontology()
    parts = ["-Ontology Definitions and Examples from Knowledge Graph-\n"]

    # Entity type definitions
    parts.append("ENTITY TYPE DEFINITIONS:")
    for et in ontology.get("entity_types", []):
        if isinstance(et, dict):
            name = et.get("name", "")
            desc = et.get("description", "")
            examples = et.get("examples", [])
            line = f"  {name}: {desc}" if desc else f"  {name}"
            if examples:
                line += f" (examples: {', '.join(examples[:3])})"
            parts.append(line)
        else:
            parts.append(f"  {et}")

    parts.append("")

    # Relationship type definitions
    parts.append("RELATIONSHIP TYPE DEFINITIONS:")
    for rt in ontology.get("relationship_types", []):
        if isinstance(rt, dict):
            name = rt.get("name", "")
            desc = rt.get("description", "")
            line = f"  {name}: {desc}" if desc else f"  {name}"
            parts.append(line)
        else:
            parts.append(f"  {rt}")

    parts.append("")

    # Neo4j graph examples (sample entities and relationships)
    try:
        from app.db.session import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Sample entities
            result = session.run(
                "MATCH (n:Entity) "
                "RETURN n.name AS name, n.entity_type AS type, "
                "n.description AS desc "
                "ORDER BY n.name LIMIT 15"
            )
            entities = [r.data() for r in result]
            if entities:
                parts.append("EXAMPLE ENTITIES FROM KNOWLEDGE GRAPH:")
                for e in entities:
                    name = e.get("name", "?")
                    etype = e.get("type", "?")
                    desc = (e.get("desc") or "")[:100]
                    parts.append(f"  {name} [{etype}]: {desc}")
                parts.append("")

            # Sample relationships
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.name AS src, type(r) AS rel, b.name AS tgt, "
                "r.description AS desc "
                "LIMIT 15"
            )
            rels = [r.data() for r in result]
            if rels:
                parts.append("EXAMPLE RELATIONSHIPS FROM KNOWLEDGE GRAPH:")
                for r in rels:
                    desc = (r.get("desc") or "")[:80]
                    parts.append(f"  {r['src']} --[{r['rel']}]--> {r['tgt']}: {desc}")
                parts.append("")

    except Exception:
        parts.append("(Neo4j graph examples unavailable)\n")

    return "\n".join(parts)


def write_prompt_files(prompts_dir: Path) -> None:
    """Write all prompt files to disk for GraphRAG to read."""
    prompts_dir.mkdir(parents=True, exist_ok=True)

    ontology_context = _build_ontology_context()

    files = {
        "extract_graph.txt": get_entity_extraction_prompt(ontology_context),
        "community_report.txt": get_community_report_prompt(),
        "local_search_system_prompt.txt": get_local_search_prompt(),
        "global_search_map_system_prompt.txt": get_global_search_map_prompt(),
        "global_search_reduce_system_prompt.txt": get_global_search_reduce_prompt(),
        "drift_search_system_prompt.txt": get_drift_search_prompt(),
        "basic_search_system_prompt.txt": get_basic_search_prompt(),
    }

    for filename, content in files.items():
        (prompts_dir / filename).write_text(content)
