# Docling-Graph Full Ontology Extraction + NER Removal

**Date**: 2026-03-16
**Status**: Approved

## Problem

The docling-graph service currently picks only the first template (DOCUMENT) from the 45 built ontology templates, producing useless DOCUMENT_* entities instead of military-relevant ones. The regex NER service (`app/services/ner.py`) is dead code — all entity extraction runs through docling-graph.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Template strategy | Focused batches (5 groups) | Better extraction quality from 8B LLM with smaller schemas |
| Group definition | Hardcoded in `templates.py` | Matches ontology layers directly, simple |
| Relationship extraction | Separate final pass with all entities as context | Cross-group relationships captured cleanly |
| Batching location | Pipeline task (`derive_ontology_graph`) | Per-batch retry control, shorter individual requests |
| API approach | Template group selection via request parameter | Minimal API change, service owns template logic |
| NER removal | Full deletion | Dead code, no fallback needed |

## Ontology Groups

| Group | Entity Types | Count |
|-------|-------------|-------|
| `reference` | DOCUMENT, SECTION, FIGURE, TABLE, SPREADSHEET, ASSERTION | 6 |
| `equipment` | PLATFORM, RADAR_SYSTEM, MISSILE_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM, ELECTRONIC_WARFARE_SYSTEM, FIRE_CONTROL_SYSTEM, INTEGRATED_AIR_DEFENSE_SYSTEM, LAUNCHER_SYSTEM, WEAPON_SYSTEM | 9 |
| `rf_signal` | FREQUENCY_BAND, RF_EMISSION, WAVEFORM, MODULATION, RF_SIGNATURE, SCAN_PATTERN, ANTENNA, TRANSMITTER, RECEIVER, IF_AMPLIFIER, SIGNAL_PROCESSING_CHAIN, SEEKER | 12 |
| `weapon` | GUIDANCE_METHOD, MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS, PROPULSION_STACK, PROPULSION_STAGE, SUBSYSTEM, COMPONENT | 7 |
| `operational` | CAPABILITY, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE, FORCE_STRUCTURE, EQUIPMENT_SYSTEM, ASSEMBLY, SPECIFICATION, STANDARD, PROCEDURE, FAILURE_MODE, TEST_EVENT | 11 |

## Component Changes

### 1. Docling-Graph Service (`docker/docling-graph/`)

#### `templates.py`

- Add `GROUP_MAP: dict[str, list[str]]` — hardcoded mapping of group name to entity type names.
- `build_templates()` returns `dict[str, type[BaseModel]]` where each key is a group name and the value is a single combined Pydantic model. Each combined model has optional list fields for every entity type in the group (e.g., `radar_system: list[RadarSystemModel] | None = None`).
- Individual per-entity-type models are still built internally as building blocks.

#### `prompts.py` (new file)

Group-specific system prompts providing domain context to the LLM:

- **reference**: "Extract document structure elements: sections, figures, tables, and assertions with their metadata."
- **equipment**: "Extract military equipment systems: platforms (vehicles, aircraft, vessels), radar systems, missile systems, air defense artillery, electronic warfare systems, fire control systems, IADS, launchers, and weapon systems. Look for system designations (AN/XXX-YY), nomenclature, and operational characteristics."
- **rf_signal**: "Extract electromagnetic and RF signal characteristics: frequency bands, RF emissions, waveforms (LFM, NLFM, Barker, Costas), modulation types, RF signatures, scan patterns, antennas, transmitters, receivers, IF amplifiers, signal processing chains, and seekers. Pay attention to specific frequency values, PRF/PRI, pulse durations, and ERP."
- **weapon**: "Extract weapon and missile subsystem details: guidance methods (SARH, ARH, IIR, GPS/INS), missile performance parameters (range, altitude, speed), physical characteristics (dimensions, mass), propulsion stages, subsystems, and components."
- **operational**: "Extract operational and capability information: functional capabilities, radar performance metrics, engagement timelines, force structure/units, equipment system designations, assemblies, specifications, standards (MIL-STD/MIL-DTL), procedures, failure modes, and test events."
- **relationships**: "Given the following entities extracted from a military technical document, identify relationships between them. Focus on: hierarchy (PART_OF, CONTAINS, HAS_SUBSYSTEM), installation (INSTALLED_ON, DEPLOYED_ON), functional links (OPERATES_IN_BAND, USES_WAVEFORM, EMITS), and tactical associations (CUES, GUIDES, TRACKS, ENGAGES)."

#### `schemas.py`

- `ExtractionRequest` gains:
  - `template_group: str | None = None` — which entity group to extract
  - `mode: str = "entities"` — `"entities"` or `"relationships"`
  - `entities_context: list[dict] | None = None` — for relationship pass, list of `{name, entity_type}` from prior entity passes
- Response schema unchanged.

#### `main.py`

- `_run_extraction()` accepts `template_group` and `mode` parameters.
- Selects the correct combined template from `_templates[template_group]`.
- Selects the correct system prompt from `prompts.py`.
- For `mode="relationships"`: injects discovered entity list into the prompt context alongside the relationship type definitions from the ontology.
- Backward compatibility: if `template_group` is None, uses the first template (existing behavior for any legacy callers).

### 2. Pipeline Client (`app/services/docling_graph_service.py`)

- `extract_graph()` gains optional `template_group`, `mode`, and `entities_context` parameters.
- These are passed through in the HTTP POST body.

### 3. Pipeline Task (`app/workers/pipeline.py` — `derive_ontology_graph`)

Extraction loop replaces single call:

```python
all_entities = []
for group in ["reference", "equipment", "rf_signal", "weapon", "operational"]:
    result = extract_graph(text, doc_id, template_group=group, mode="entities")
    all_entities.extend(result["entities"])

# Final relationship pass with all discovered entities as context
entities_context = [{"name": e["name"], "entity_type": e["entity_type"]} for e in all_entities]
result = extract_graph(text, doc_id, mode="relationships", entities_context=entities_context)
all_relationships = result["relationships"]
```

Confidence filtering and Neo4j import process the merged lists unchanged.

### 4. NER Removal

- Delete `app/services/ner.py`
- Delete `tests/unit/test_ner.py`
- No pipeline imports to remove (NER is not called)

### 5. Concurrency & Timeout

- Each document holds the Redis concurrency permit for all 6 HTTP calls (existing gate).
- `DOCLING_GRAPH_TIMEOUT` applies per-call (each ~30-45s). Total pipeline task time covered by existing `GRAPH_SOFT_TIME_LIMIT=600` / `GRAPH_TIME_LIMIT=900`.
- If 6 × 45s = 270s exceeds soft limit, bump `GRAPH_SOFT_TIME_LIMIT` to 900 and `GRAPH_TIME_LIMIT` to 1200.

## Data Flow

```
Document text
    │
    ├─ POST /extract {group="reference", mode="entities"}  → entities[]
    ├─ POST /extract {group="equipment", mode="entities"}  → entities[]
    ├─ POST /extract {group="rf_signal", mode="entities"}  → entities[]
    ├─ POST /extract {group="weapon", mode="entities"}     → entities[]
    ├─ POST /extract {group="operational", mode="entities"} → entities[]
    │
    └─ POST /extract {mode="relationships", entities_context=[...all entities...]}
           → relationships[]
    │
    ▼
Confidence filter → Neo4j batch import → DocumentGraphExtraction record
```

## Testing

- Update `test_pipeline_graph.py` to mock 6 calls instead of 1.
- Update docling-graph service unit tests for new request fields.
- Mock mode in docling-graph should respect `template_group` and return group-appropriate mock entities.
- Existing integration tests validate Neo4j import unchanged.

## Files Changed

| File | Action |
|------|--------|
| `docker/docling-graph/app/templates.py` | Modify — add GROUP_MAP, return combined models per group |
| `docker/docling-graph/app/prompts.py` | Create — group-specific LLM prompts |
| `docker/docling-graph/app/schemas.py` | Modify — add template_group, mode, entities_context fields |
| `docker/docling-graph/app/main.py` | Modify — route by group/mode, inject prompts |
| `app/services/docling_graph_service.py` | Modify — pass new fields to HTTP call |
| `app/workers/pipeline.py` | Modify — 6-call extraction loop |
| `app/services/ner.py` | Delete |
| `tests/unit/test_ner.py` | Delete |
| `app/config.py` | Modify — bump time limits if needed |
