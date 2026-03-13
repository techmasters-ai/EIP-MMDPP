# Comprehensive Quality Pass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Improve ontology extraction precision, relationship quality, and retrieval relevance across the full ingest+retrieval pipeline.

**Architecture:** Fix extraction prompt, property validation, model defaults, and the retrieval scoring pipeline. Add a cross-encoder reranker service. Wire the existing structure-aware chunker into the live pipeline. All changes are backward-compatible with re-ingest.

**Tech Stack:** Python 3.11, FastAPI, Celery, sentence-transformers, cross-encoders, Qdrant, Neo4j, Ollama

---

### Task 1: Enrich ontology property definitions in base_v1.yaml

**Files:**
- Modify: `ontology/base_v1.yaml`
- Test: `tests/unit/test_ontology_templates.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_ontology_templates.py`:

```python
def test_all_properties_have_descriptions():
    """Every property in the ontology YAML must have a description."""
    from app.services.ontology_templates import load_ontology
    ontology = load_ontology()
    missing = []
    for et in ontology.get("entity_types", []):
        props = et.get("properties", {}).get("properties", {})
        for prop_name, prop_def in props.items():
            if not prop_def.get("description"):
                missing.append(f"{et['name']}.{prop_name}")
    assert not missing, f"Properties missing descriptions: {missing}"


def test_structured_properties_have_examples():
    """Properties with patterns should have examples."""
    from app.services.ontology_templates import load_ontology
    ontology = load_ontology()
    missing = []
    for et in ontology.get("entity_types", []):
        props = et.get("properties", {}).get("properties", {})
        for prop_name, prop_def in props.items():
            if prop_def.get("pattern") and not prop_def.get("example"):
                missing.append(f"{et['name']}.{prop_name}")
    assert not missing, f"Pattern properties missing examples: {missing}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ontology_templates.py::test_all_properties_have_descriptions -v`
Expected: FAIL — many properties lack descriptions

**Step 3: Update `ontology/base_v1.yaml` with descriptions, examples, and patterns**

Add `description`, `example`, and `pattern` to every property across all 12 entity types. Key examples:

```yaml
# COMPONENT properties:
name: {type: string, description: "Common name of the component", example: "Traveling Wave Tube"}
part_number: {type: string, description: "Manufacturer-assigned part number (alphanumeric with dashes)", example: "PN-12345-A", pattern: "^[A-Z0-9][A-Z0-9\\-/]{2,20}$"}
nsn: {type: string, description: "National Stock Number in NNNN-NN-NNN-NNNN format", example: "5961-01-234-5678", pattern: "^\\d{4}-\\d{2}-\\d{3}-\\d{4}$"}
cage_code: {type: string, description: "5-character Commercial and Government Entity code identifying the manufacturer", example: "1ABC3", pattern: "^[A-Z0-9]{5}$"}
material: {type: string, description: "Primary material composition of the component", example: "Aluminum 7075-T6"}
weight_kg: {type: number, description: "Weight of the component in kilograms", example: 2.5}

# STANDARD properties:
designation: {type: string, description: "Official standard designation number (e.g. MIL-STD-1553B)", example: "MIL-STD-1553B", pattern: "^MIL-[A-Z]+-\\d+[A-Z]?"}
title: {type: string, description: "Full title of the standard document", example: "Digital Time Division Command/Response Multiplex Data Bus"}
issuing_org: {type: string, description: "Organization that published the standard", example: "Department of Defense"}
version: {type: string, description: "Revision letter or version number", example: "B"}
supersedes: {type: string, description: "Designation of the standard this one replaces", example: "MIL-STD-1553A"}

# SPECIFICATION properties:
parameter: {type: string, description: "Name of the measured parameter (e.g. max_range, operating_temperature)", example: "max_range"}
value: {type: string, description: "Numeric value or range of the measurement", example: "150"}
unit: {type: string, description: "Unit of measurement (SI or military standard)", example: "km"}
condition: {type: string, description: "Operating conditions under which the spec applies", example: "sea level, standard atmosphere"}
source_document: {type: string, description: "Document where this specification is defined", example: "TM 9-1425-386-12"}

# EQUIPMENT_SYSTEM properties:
name: {type: string, description: "Common name or designation of the system", example: "Patriot PAC-3"}
designation: {type: string, description: "Military AN/ designation or program designation", example: "AN/MPQ-65", pattern: "^AN/[A-Z]{3}-\\d+"}
program_office: {type: string, description: "Managing program office", example: "PEO Missiles and Space"}
status: {type: string, description: "Current lifecycle status", enum: [DEVELOPMENTAL, OPERATIONAL, RETIRED, PROTOTYPE]}
prime_contractor: {type: string, description: "Lead contractor organization", example: "Lockheed Martin"}
service_branch: {type: string, description: "Military branch operating the system", example: "U.S. Army"}

# ORGANIZATION properties:
name: {type: string, description: "Full name of the organization", example: "Raytheon Missiles & Defense"}
type: {type: string, description: "Category of organization", enum: [PRIME_CONTRACTOR, SUBCONTRACTOR, PROGRAM_OFFICE, MILITARY_BRANCH, GOVERNMENT_AGENCY]}
cage_code: {type: string, description: "5-character CAGE code for the organization", example: "58064", pattern: "^[A-Z0-9]{5}$"}
location: {type: string, description: "Primary facility location", example: "Tucson, AZ"}

# SUBSYSTEM properties:
name: {type: string, description: "Name of the subsystem", example: "Guidance Section"}
function: {type: string, description: "Primary functional role of the subsystem", example: "Terminal phase target tracking and guidance"}
part_number: {type: string, description: "Subsystem-level part or drawing number", example: "GS-PAC3-001"}

# ASSEMBLY properties:
name: {type: string, description: "Name of the assembly unit", example: "Antenna Feed Assembly"}
assembly_number: {type: string, description: "Assembly drawing or identification number", example: "ASM-7891-A"}

# CAPABILITY properties:
name: {type: string, description: "Name of the functional capability", example: "Terminal Phase Guidance"}
category: {type: string, description: "Capability domain or category", example: "Guidance and Navigation"}
trl: {type: integer, description: "Technology Readiness Level on a scale of 1-9", example: 7}

# DOCUMENT properties:
title: {type: string, description: "Full title of the document", example: "Operator Manual for Patriot Missile System"}
document_number: {type: string, description: "Official document identifier or TM number", example: "TM 9-1425-386-12"}
revision: {type: string, description: "Document revision identifier", example: "Rev C"}
classification: {type: string, description: "Security classification level", example: "UNCLASSIFIED"}
issuing_org: {type: string, description: "Organization that published the document", example: "U.S. Army TACOM"}
date: {type: string, description: "Publication or revision date (YYYY-MM-DD)", example: "2023-06-15"}

# PROCEDURE properties:
name: {type: string, description: "Name or title of the procedure", example: "Radar Antenna Alignment Procedure"}
type: {type: string, description: "Category of procedure", enum: [MAINTENANCE, OPERATIONAL, TEST, CALIBRATION, INSPECTION]}
periodicity: {type: string, description: "How often the procedure must be performed", example: "Semi-annual"}
skill_level: {type: string, description: "Required maintenance skill level", example: "20C (Patriot Repairer)"}

# FAILURE_MODE properties:
name: {type: string, description: "Short name of the failure mode", example: "TWT Power Degradation"}
description: {type: string, description: "Detailed description of how the failure manifests", example: "Gradual loss of transmit power due to cathode erosion"}
fmeca_severity: {type: integer, description: "MIL-STD-1629 severity category (1=catastrophic, 4=minor)", example: 2}
detection_method: {type: string, description: "How this failure is detected", example: "BIT fault code 47, power output below threshold"}

# TEST_EVENT properties:
name: {type: string, description: "Name or designation of the test event", example: "FET-10 Flight Test"}
date: {type: string, description: "Date the test was conducted (YYYY-MM-DD)", example: "2024-03-15"}
location: {type: string, description: "Test range or facility location", example: "White Sands Missile Range, NM"}
test_type: {type: string, description: "Category of test event", enum: [DT, OT, IOT, LFT, DEVELOPMENTAL]}
outcome: {type: string, description: "Overall test result", enum: [PASS, FAIL, PARTIAL, INCONCLUSIVE]}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ontology_templates.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ontology/base_v1.yaml tests/unit/test_ontology_templates.py
git commit -m "feat: enrich ontology property definitions with descriptions, examples, patterns"
```

---

### Task 2: Add validation matrix to ontology

**Files:**
- Modify: `ontology/base_v1.yaml`
- Modify: `app/services/ontology_templates.py:224-239` — fix `load_validation_matrix` format
- Test: `tests/unit/test_ontology_templates.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_ontology_templates.py`:

```python
def test_validation_matrix_loads():
    """Validation matrix should load all 10 relationship types."""
    from app.services.ontology_templates import load_validation_matrix
    matrix = load_validation_matrix()
    assert len(matrix) > 0, "Validation matrix is empty"
    # Check a known valid triple
    assert ("SUBSYSTEM", "IS_SUBSYSTEM_OF", "EQUIPMENT_SYSTEM") in matrix


def test_validation_matrix_covers_all_relationships():
    """Every relationship type in the ontology should appear in the validation matrix."""
    from app.services.ontology_templates import load_ontology, load_validation_matrix
    ontology = load_ontology()
    matrix = load_validation_matrix()
    rel_names = {rt["name"] for rt in ontology.get("relationship_types", [])}
    matrix_rels = {triple[1] for triple in matrix}
    missing = rel_names - matrix_rels
    assert not missing, f"Relationship types missing from validation matrix: {missing}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ontology_templates.py::test_validation_matrix_loads -v`
Expected: FAIL — matrix is empty

**Step 3: Add validation_matrix to `base_v1.yaml` and fix `load_validation_matrix`**

Add to `ontology/base_v1.yaml` at the end:

```yaml
validation_matrix:
  - source_types: [SUBSYSTEM]
    relationship: IS_SUBSYSTEM_OF
    target_types: [EQUIPMENT_SYSTEM]
  - source_types: [ASSEMBLY, SUBSYSTEM, EQUIPMENT_SYSTEM]
    relationship: CONTAINS
    target_types: [COMPONENT, ASSEMBLY]
  - source_types: [EQUIPMENT_SYSTEM, SUBSYSTEM]
    relationship: IMPLEMENTS
    target_types: [CAPABILITY]
  - source_types: [COMPONENT, SUBSYSTEM, EQUIPMENT_SYSTEM, ASSEMBLY]
    relationship: MEETS_STANDARD
    target_types: [STANDARD]
  - source_types: [COMPONENT, SUBSYSTEM, EQUIPMENT_SYSTEM]
    relationship: SPECIFIED_BY
    target_types: [SPECIFICATION]
  - source_types: [EQUIPMENT_SYSTEM, SUBSYSTEM, COMPONENT, ASSEMBLY, SPECIFICATION, CAPABILITY, STANDARD, ORGANIZATION, PROCEDURE, FAILURE_MODE, TEST_EVENT]
    relationship: DESCRIBED_IN
    target_types: [DOCUMENT]
  - source_types: [PROCEDURE]
    relationship: PERFORMED_BY
    target_types: [ORGANIZATION]
  - source_types: [FAILURE_MODE]
    relationship: AFFECTS
    target_types: [COMPONENT, SUBSYSTEM, ASSEMBLY]
  - source_types: [DOCUMENT, STANDARD]
    relationship: SUPERSEDES
    target_types: [DOCUMENT, STANDARD]
  - source_types: [EQUIPMENT_SYSTEM, SUBSYSTEM, COMPONENT]
    relationship: TESTED_IN
    target_types: [TEST_EVENT]
```

Update `load_validation_matrix` in `app/services/ontology_templates.py` (line 224) to expand the matrix from the list format:

```python
def load_validation_matrix(
    path: Path | None = None,
) -> set[tuple[str, str, str]]:
    """Load the ontology validation matrix as a set of (source, rel, target) triples.

    Expands source_types × target_types into individual triples.
    Returns an empty set if the ontology doesn't define a validation_matrix.
    """
    ontology = load_ontology(path)
    matrix: set[tuple[str, str, str]] = set()
    for entry in ontology.get("validation_matrix", []):
        rel = entry.get("relationship", "")
        source_types = entry.get("source_types", [])
        target_types = entry.get("target_types", [])
        if not rel:
            continue
        for src in source_types:
            for tgt in target_types:
                matrix.add((src, rel, tgt))
    return matrix
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ontology_templates.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ontology/base_v1.yaml app/services/ontology_templates.py tests/unit/test_ontology_templates.py
git commit -m "feat: add validation matrix to ontology, fix load_validation_matrix expansion"
```

---

### Task 3: Fix extraction prompt and add post-extraction validation

**Files:**
- Modify: `app/services/ontology_templates.py:126-221` — `build_extraction_prompt`
- Modify: `app/services/docling_graph_service.py:569-592` — `_validate_triples` + new `_validate_entity_types` + `_validate_properties`
- Test: `tests/unit/test_docling_graph_service.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_docling_graph_service.py`:

```python
def test_validate_entity_types_rejects_unknown():
    """Entities with types not in the ontology should be rejected."""
    from app.services.docling_graph_service import _validate_entity_types
    from app.services.ontology_templates import ExtractedEntity, DocumentExtractionResult

    extraction = DocumentExtractionResult(entities=[
        ExtractedEntity(entity_type="EQUIPMENT_SYSTEM", name="Patriot", confidence=0.9),
        ExtractedEntity(entity_type="RADAR_SYSTEM", name="AN/MPQ-65", confidence=0.9),
    ])
    result = _validate_entity_types(extraction)
    assert len(result.entities) == 1
    assert result.entities[0].entity_type == "EQUIPMENT_SYSTEM"


def test_validate_properties_rejects_misplaced_nsn():
    """An NSN-formatted value in cage_code field should be removed."""
    from app.services.docling_graph_service import _validate_properties
    from app.services.ontology_templates import ExtractedEntity, DocumentExtractionResult

    extraction = DocumentExtractionResult(entities=[
        ExtractedEntity(
            entity_type="COMPONENT",
            name="Widget",
            properties={"cage_code": "5961-01-234-5678", "nsn": "5961-01-234-5678"},
            confidence=0.9,
        ),
    ])
    result = _validate_properties(extraction)
    # cage_code should be removed (doesn't match ^[A-Z0-9]{5}$), nsn should remain
    assert "cage_code" not in result.entities[0].properties
    assert "nsn" in result.entities[0].properties


def test_prompt_uses_only_ontology_types():
    """Few-shot example in prompt should only use valid ontology types."""
    from app.services.ontology_templates import build_extraction_prompt, load_ontology
    ontology = load_ontology()
    prompt = build_extraction_prompt(ontology, "test text", few_shot=True)
    assert "RADAR_SYSTEM" not in prompt
    assert "OPERATES_IN_BAND" not in prompt
    assert "FREQUENCY_BAND" not in prompt


def test_prompt_includes_property_descriptions():
    """Prompt should include property descriptions for each entity type."""
    from app.services.ontology_templates import build_extraction_prompt, load_ontology
    ontology = load_ontology()
    prompt = build_extraction_prompt(ontology, "test text")
    # Check that descriptions appear (not just property names)
    assert "National Stock Number" in prompt
    assert "Manufacturer-assigned part number" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_docling_graph_service.py::test_validate_entity_types_rejects_unknown -v`
Expected: FAIL — `_validate_entity_types` doesn't exist yet

**Step 3: Fix the prompt**

In `app/services/ontology_templates.py`, update `build_extraction_prompt` (line 126):

1. Replace the few-shot example (line 171-175) with one using valid ontology types:
```python
    few_shot_block = ""
    if few_shot:
        few_shot_block = """
## Example
Input: "The Patriot PAC-3 missile system uses MIL-STD-1553B for internal data bus communication. The guidance section contains a Ka-band seeker operating at 35 GHz."
Output: {"entities": [{"entity_type": "EQUIPMENT_SYSTEM", "name": "Patriot PAC-3", "properties": {"name": "Patriot PAC-3"}, "confidence": 0.95}, {"entity_type": "STANDARD", "name": "MIL-STD-1553B", "properties": {"designation": "MIL-STD-1553B"}, "confidence": 0.95}, {"entity_type": "SUBSYSTEM", "name": "Guidance Section", "properties": {"name": "Guidance Section", "function": "Terminal phase target tracking and guidance"}, "confidence": 0.85}, {"entity_type": "COMPONENT", "name": "Ka-band Seeker", "properties": {"name": "Ka-band Seeker"}, "confidence": 0.80}], "relationships": [{"relationship_type": "MEETS_STANDARD", "from_name": "Patriot PAC-3", "from_type": "EQUIPMENT_SYSTEM", "to_name": "MIL-STD-1553B", "to_type": "STANDARD", "properties": {}, "confidence": 0.90}, {"relationship_type": "IS_SUBSYSTEM_OF", "from_name": "Guidance Section", "from_type": "SUBSYSTEM", "to_name": "Patriot PAC-3", "to_type": "EQUIPMENT_SYSTEM", "properties": {}, "confidence": 0.85}, {"relationship_type": "CONTAINS", "from_name": "Guidance Section", "from_type": "SUBSYSTEM", "to_name": "Ka-band Seeker", "to_type": "COMPONENT", "properties": {}, "confidence": 0.80}]}
"""
```

2. Include property descriptions in the entity type listing (replace lines 144-156):
```python
    entity_descriptions = []
    for et in ontology.get("entity_types", []):
        props = et.get("properties", {}).get("properties", {})
        if compact_ontology:
            prop_preview = ", ".join(list(props.keys())[:max_properties_per_entity]) or "none"
            if len(props) > max_properties_per_entity:
                prop_preview += ", ..."
            entity_descriptions.append(f"  - {et['name']} (props: {prop_preview})")
        else:
            prop_details = []
            for pname, pdef in props.items():
                desc = pdef.get("description", "")
                example = pdef.get("example", "")
                detail = f"{pname}"
                if desc:
                    detail += f" ({desc})"
                if example:
                    detail += f" [e.g. {example}]"
                prop_details.append(detail)
            entity_descriptions.append(
                f"  - {et['name']}: {et.get('description', '')}\n"
                f"    Properties: {', '.join(prop_details)}"
            )
```

3. Add explicit instruction to the prompt (after the JSON schema block, before the document text):
```
Only use entity types and relationship types from the lists above. Do not invent new types.
Place each extracted value in the correct property field based on its description.
```

**Step 4: Add post-extraction validation functions**

In `app/services/docling_graph_service.py`, add after `_validate_triples` (after line 592):

```python
def _validate_entity_types(
    extraction: DocumentExtractionResult,
    ontology_path=None,
) -> DocumentExtractionResult:
    """Remove entities whose entity_type is not in the ontology."""
    from app.services.ontology_templates import build_entity_type_names
    valid_types = set(build_entity_type_names(load_ontology(ontology_path)))
    valid_entities = []
    for entity in extraction.entities:
        if entity.entity_type in valid_types:
            valid_entities.append(entity)
        else:
            logger.warning(
                "Rejected entity with unknown type: %s (type=%s)",
                entity.name, entity.entity_type,
            )
    return DocumentExtractionResult(
        entities=valid_entities,
        relationships=extraction.relationships,
    )


def _validate_properties(
    extraction: DocumentExtractionResult,
    ontology_path=None,
) -> DocumentExtractionResult:
    """Validate entity property values against ontology patterns.

    Removes individual property values that don't match their declared pattern.
    Does not remove the entity — just the misplaced value.
    """
    import re as _re
    ontology = load_ontology(ontology_path)
    # Build pattern lookup: {entity_type: {prop_name: compiled_pattern}}
    patterns: dict[str, dict[str, _re.Pattern]] = {}
    for et in ontology.get("entity_types", []):
        type_patterns = {}
        for pname, pdef in et.get("properties", {}).get("properties", {}).items():
            if pdef.get("pattern"):
                type_patterns[pname] = _re.compile(pdef["pattern"])
        if type_patterns:
            patterns[et["name"]] = type_patterns

    validated_entities = []
    for entity in extraction.entities:
        type_patterns = patterns.get(entity.entity_type, {})
        if not type_patterns:
            validated_entities.append(entity)
            continue
        clean_props = {}
        for k, v in entity.properties.items():
            pattern = type_patterns.get(k)
            if pattern and isinstance(v, str) and not pattern.match(v):
                logger.warning(
                    "Property validation failed: %s.%s = '%s' (pattern: %s)",
                    entity.entity_type, k, v, pattern.pattern,
                )
                continue
            clean_props[k] = v
        validated_entities.append(ExtractedEntity(
            entity_type=entity.entity_type,
            name=entity.name,
            properties=clean_props,
            confidence=entity.confidence,
        ))

    return DocumentExtractionResult(
        entities=validated_entities,
        relationships=extraction.relationships,
    )
```

Wire them into `_extract_single_pass` (line 87-88), adding after `_validate_triples`:

```python
            extraction = _validate_triples(extraction, ontology_path)
            extraction = _validate_entity_types(extraction, ontology_path)
            extraction = _validate_properties(extraction, ontology_path)
```

**Step 4b: Also filter relationships whose endpoint types were rejected**

After `_validate_entity_types`, the existing `_build_networkx_graph` already handles missing endpoints by auto-creating nodes at confidence 0.3. But we want to reject relationships where the *type* itself is invalid. Add a relationship type check inside `_validate_entity_types`:

```python
    # Also filter relationships with invalid types
    from app.services.ontology_templates import build_relationship_type_names
    valid_rel_types = set(build_relationship_type_names(load_ontology(ontology_path)))
    valid_rels = []
    for rel in extraction.relationships:
        if rel.relationship_type in valid_rel_types:
            valid_rels.append(rel)
        else:
            logger.warning(
                "Rejected relationship with unknown type: %s -[%s]-> %s",
                rel.from_name, rel.relationship_type, rel.to_name,
            )
    return DocumentExtractionResult(
        entities=valid_entities,
        relationships=valid_rels,
    )
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_docling_graph_service.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/services/ontology_templates.py app/services/docling_graph_service.py tests/unit/test_docling_graph_service.py
git commit -m "feat: fix extraction prompt, add entity/property/relationship type validation"
```

---

### Task 4: Align NER entity types with ontology

**Files:**
- Modify: `app/services/ner.py`
- Modify: `tests/unit/test_ner.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_ner.py`:

```python
def test_all_entity_types_are_valid_ontology_types():
    """Every entity_type produced by NER must exist in the ontology."""
    from app.services.ner import extract_entities
    from app.services.ontology_templates import build_entity_type_names
    valid_types = set(build_entity_type_names())

    # Text that triggers multiple NER patterns
    text = (
        "ELNOT FIRE HAWK operates in X-band at 9.4 GHz. "
        "DIEQP ABC-123 uses LFM waveform with PRI 1500 μs. "
        "Semi-active radar guidance. Track-while-scan mode. "
        "Barrage jamming countermeasure."
    )
    entities = extract_entities(text)
    invalid = [(e.name, e.entity_type) for e in entities if e.entity_type not in valid_types]
    assert not invalid, f"NER produced invalid entity types: {invalid}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ner.py::test_all_entity_types_are_valid_ontology_types -v`
Expected: FAIL — `RADAR_SYSTEM`, `RF_EMISSION`, `FREQUENCY_BAND`, `WAVEFORM`, `SCAN_PATTERN`, `GUIDANCE_METHOD`, `ELECTRONIC_WARFARE_SYSTEM` are not in the ontology

**Step 3: Map NER types to ontology types**

In `app/services/ner.py`, update the entity_type assignments:

| Old NER type | New ontology type | Rationale |
|---|---|---|
| `RADAR_SYSTEM` | `EQUIPMENT_SYSTEM` | Radar systems are equipment systems |
| `RF_EMISSION` | `SPECIFICATION` | Frequency values are specifications |
| `FREQUENCY_BAND` | `SPECIFICATION` | Band designations are specifications |
| `WAVEFORM` | `SPECIFICATION` | Waveform params are specifications |
| `SCAN_PATTERN` | `CAPABILITY` | Radar modes are capabilities |
| `GUIDANCE_METHOD` | `CAPABILITY` | Guidance types are capabilities |
| `ELECTRONIC_WARFARE_SYSTEM` | `CAPABILITY` | EW techniques are capabilities |

Update each `EntityCandidate` constructor call:
- Line 377-384: `entity_type="RADAR_SYSTEM"` → `entity_type="EQUIPMENT_SYSTEM"` (ELNOT)
- Line 389-397: `entity_type="RADAR_SYSTEM"` → `entity_type="EQUIPMENT_SYSTEM"` (DIEQP)
- Line 400-411: `entity_type="RF_EMISSION"` → `entity_type="SPECIFICATION"` (frequency)
- Line 414-424: `entity_type="FREQUENCY_BAND"` → `entity_type="SPECIFICATION"` (freq band)
- Line 427-439: `entity_type="WAVEFORM"` → `entity_type="SPECIFICATION"` (PRI/PRF)
- Line 442-452: `entity_type="SCAN_PATTERN"` → `entity_type="CAPABILITY"` (radar modes)
- Line 455-465: `entity_type="WAVEFORM"` → `entity_type="SPECIFICATION"` (waveform families)
- Line 468-478: `entity_type="GUIDANCE_METHOD"` → `entity_type="CAPABILITY"` (guidance)
- Line 481-491: `entity_type="ELECTRONIC_WARFARE_SYSTEM"` → `entity_type="CAPABILITY"` (EW)

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/ner.py tests/unit/test_ner.py
git commit -m "fix: align NER entity types with ontology definitions"
```

---

### Task 5: Update model defaults and config

**Files:**
- Modify: `app/config.py:93,102,90`
- Modify: `env.example`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_config.py`:

```python
def test_new_reranker_config_defaults():
    """Reranker config fields should exist with correct defaults."""
    from app.config import Settings
    s = Settings(app_env="test")
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.reranker_device == "cpu"
    assert s.reranker_enabled is True
    assert s.reranker_top_n == 20
    assert s.retrieval_min_score_threshold == 0.25


def test_updated_extraction_defaults():
    """Extraction model defaults should reflect the upgraded values."""
    from app.config import Settings
    s = Settings(app_env="test")
    assert s.docling_graph_model == "llama3.1:8b"
    assert s.docling_graph_max_tokens == 2048
    assert s.ollama_num_ctx == 16384
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_new_reranker_config_defaults -v`
Expected: FAIL — fields don't exist

**Step 3: Update `app/config.py`**

Add new fields after line 123 (after `qdrant_timeout_seconds`):

```python
    # Reranker (cross-encoder for retrieval re-scoring)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # cpu | cuda
    reranker_enabled: bool = True
    reranker_top_n: int = 20

    # Minimum cosine similarity threshold (below this, results are dropped)
    retrieval_min_score_threshold: float = 0.25
```

Update existing defaults:
- Line 93: `docling_graph_model: str = "llama3.2"` → `"llama3.1:8b"`
- Line 102: `docling_graph_max_tokens: int = 1200` → `2048`
- Line 90: `ollama_num_ctx: int = 8192` → `16384`

Update `env.example` — add new section after retrieval scoring:

```bash
# ---------------------------------------------------------------------------
# Reranker (cross-encoder for retrieval re-scoring)
# ---------------------------------------------------------------------------
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu                     # cpu | cuda
RERANKER_ENABLED=true
RERANKER_TOP_N=20

# Minimum cosine similarity for results (below this, results are dropped)
RETRIEVAL_MIN_SCORE_THRESHOLD=0.25
```

Update existing values in `env.example`:
- `DOCLING_GRAPH_MODEL=llama3.1:8b`
- `OLLAMA_NUM_CTX=16384`

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py env.example tests/unit/test_config.py
git commit -m "feat: add reranker config, update extraction model defaults"
```

---

### Task 6: Fix BGE query prefix (asymmetric retrieval)

**Files:**
- Modify: `app/services/embedding.py:40-65`
- Modify: `app/api/v1/retrieval.py:266` — pass `query=True`
- Test: `tests/unit/test_extraction.py` (or new test file)

**Step 1: Write the failing test**

Add to `tests/unit/test_extraction.py` (or a new file if this doesn't fit):

```python
def test_bge_query_prefix_differs_from_passage():
    """BGE query embedding should use a different prefix than passage embedding."""
    from unittest.mock import patch, MagicMock
    from app.services.embedding import embed_texts

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 1024])

    with patch("app.services.embedding._get_text_model", return_value=mock_model):
        with patch("app.services.embedding.settings") as mock_settings:
            mock_settings.text_embedding_model = "BAAI/bge-large-en-v1.5"

            # Passage embedding
            embed_texts(["test text"], query=False)
            passage_call = mock_model.encode.call_args_list[-1]
            assert passage_call[0][0][0].startswith("Represent this sentence:")

            # Query embedding
            embed_texts(["test text"], query=True)
            query_call = mock_model.encode.call_args_list[-1]
            assert query_call[0][0][0].startswith("Represent this query")
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `embed_texts` doesn't accept `query` parameter

**Step 3: Update `embed_texts` in `app/services/embedding.py`**

```python
def embed_texts(texts: list[str], batch_size: int = 64, *, query: bool = False) -> list[list[float]]:
    """Embed a list of text strings. Returns list of float vectors.

    Args:
        texts: List of input strings.
        batch_size: Batch size for GPU inference.
        query: If True, use the BGE query prefix (for search queries).
               If False, use the passage prefix (for indexing documents).
    """
    if not texts:
        return []

    model = _get_text_model()

    if "bge" in settings.text_embedding_model.lower():
        if query:
            texts = [f"Represent this query for searching relevant passages: {t}" for t in texts]
        else:
            texts = [f"Represent this sentence: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
```

Update `embed_query` to use `query=True`:

```python
def embed_query(query: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([query], query=True)[0]
```

Update `_text_vector_search` in `app/api/v1/retrieval.py` (line 266):

```python
    query_embedding = embed_texts([body.query_text], query=True)[0]
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_extraction.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/embedding.py app/api/v1/retrieval.py tests/unit/test_extraction.py
git commit -m "fix: use correct BGE query prefix for asymmetric retrieval"
```

---

### Task 7: Create reranker service

**Files:**
- Create: `app/services/reranker.py`
- Test: `tests/unit/test_reranker.py`

**Step 1: Write the failing test**

Create `tests/unit/test_reranker.py`:

```python
from unittest.mock import patch, MagicMock


def test_rerank_returns_sorted_by_score():
    """Reranker should return candidates sorted by cross-encoder score."""
    from app.services.reranker import rerank

    candidates = [
        {"chunk_id": "a", "content_text": "Patriot missile system overview"},
        {"chunk_id": "b", "content_text": "Weather forecast for tomorrow"},
        {"chunk_id": "c", "content_text": "PAC-3 guidance section specifications"},
    ]

    mock_model = MagicMock()
    # b is irrelevant, a and c are relevant
    mock_model.predict.return_value = [0.8, 0.1, 0.95]

    with patch("app.services.reranker._get_reranker_model", return_value=mock_model):
        result = rerank("Patriot PAC-3 guidance", candidates, top_k=2)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "c"  # highest score
    assert result[1]["chunk_id"] == "a"


def test_rerank_disabled_returns_unchanged():
    """When reranker is disabled, return input unchanged."""
    from app.services.reranker import rerank

    candidates = [{"chunk_id": "a", "content_text": "text", "score": 0.5}]

    with patch("app.services.reranker.settings") as mock_settings:
        mock_settings.reranker_enabled = False
        result = rerank("query", candidates, top_k=10)

    assert result == candidates
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reranker.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Create `app/services/reranker.py`**

```python
"""Cross-encoder reranker for retrieval result re-scoring.

Uses BAAI/bge-reranker-v2-m3 (or configurable model) to re-score
the top-N retrieval candidates against the actual query text.

Runs on CPU by default (RERANKER_DEVICE=cpu). Set to 'cuda' for GPU.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_reranker_model():
    """Load and cache the cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder

    model_name = settings.reranker_model
    device = settings.reranker_device
    logger.info("Loading reranker model: %s (device=%s)", model_name, device)
    model = CrossEncoder(model_name, device=device)
    logger.info("Reranker model loaded")
    return model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Re-score candidates using cross-encoder and return top_k sorted by relevance.

    Args:
        query: The search query text.
        candidates: List of result dicts, each must have 'content_text'.
        top_k: Number of top results to return.

    Returns:
        Re-scored and sorted candidates (top_k).
    """
    if not settings.reranker_enabled:
        return candidates

    if not candidates or not query:
        return candidates[:top_k]

    # Filter candidates that have text to score
    scoreable = [(i, c) for i, c in enumerate(candidates) if c.get("content_text")]
    unscorable = [c for c in candidates if not c.get("content_text")]

    if not scoreable:
        return candidates[:top_k]

    model = _get_reranker_model()

    # Build query-document pairs
    pairs = [(query, c["content_text"]) for _, c in scoreable]
    scores = model.predict(pairs)

    # Attach scores and sort
    scored = []
    for (orig_idx, candidate), score in zip(scoreable, scores):
        candidate = dict(candidate)  # copy to avoid mutation
        candidate["reranker_score"] = float(score)
        scored.append(candidate)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)

    # Append unscorable items at the end
    result = scored + unscorable
    return result[:top_k]
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_reranker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/reranker.py tests/unit/test_reranker.py
git commit -m "feat: add cross-encoder reranker service (CPU default, GPU optional)"
```

---

### Task 8: Wire reranker into retrieval pipeline

**Files:**
- Modify: `app/api/v1/retrieval.py` — add reranker call after fusion scoring
- Test: `tests/unit/test_retrieval_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_retrieval_pipeline.py`:

```python
def test_reranker_applied_after_fusion():
    """Retrieval pipeline should call reranker on final results."""
    # This is an integration-level test — verify the reranker is wired in
    # by checking that the import and call exist
    import ast
    import inspect
    from app.api.v1 import retrieval
    source = inspect.getsource(retrieval)
    assert "rerank" in source, "reranker not wired into retrieval module"
```

**Step 2: Wire reranker into `_multi_modal_pipeline`**

In `app/api/v1/retrieval.py`, after the final sort and before returning results in `_multi_modal_pipeline`, add:

```python
    # Re-rank top candidates using cross-encoder
    from app.services.reranker import rerank as cross_encoder_rerank
    from app.config import get_settings as _gs
    _s = _gs()
    if _s.reranker_enabled and body.query_text:
        rerank_input = [
            {
                "chunk_id": str(r.chunk_id),
                "content_text": r.content_text,
                "score": r.score,
                "artifact_id": r.artifact_id,
                "document_id": r.document_id,
                "modality": r.modality,
                "page_number": r.page_number,
                "classification": r.classification,
            }
            for r in results[:_s.reranker_top_n]
        ]
        reranked = cross_encoder_rerank(body.query_text, rerank_input, top_k=body.top_k)
        # Rebuild QueryResultItems from reranked dicts
        results = [
            QueryResultItem(
                chunk_id=r["chunk_id"],
                artifact_id=r.get("artifact_id"),
                document_id=r.get("document_id"),
                score=r.get("reranker_score", r.get("score", 0.0)),
                modality=r.get("modality", "text"),
                content_text=r.get("content_text"),
                page_number=r.get("page_number"),
                classification=r.get("classification", "UNCLASSIFIED"),
            )
            for r in reranked
        ]
```

Also wire it into `_text_vector_search` for the `basic` strategy (same pattern, after diversification).

**Step 3: Run tests**

Run: `pytest tests/unit/test_retrieval_pipeline.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add app/api/v1/retrieval.py tests/unit/test_retrieval_pipeline.py
git commit -m "feat: wire cross-encoder reranker into retrieval pipeline"
```

---

### Task 9: Wire structure-aware chunker into pipeline

**Files:**
- Modify: `app/workers/pipeline.py:990-1040` — replace `chunk_text` with `structure_aware_chunk`
- Test: `tests/unit/test_chunking.py`

**Step 1: Verify existing chunking tests pass**

Run: `pytest tests/unit/test_chunking.py -v`

**Step 2: Update `derive_text_chunks_and_embeddings`**

In `app/workers/pipeline.py`, replace the chunking section (lines 990-1040):

```python
    from app.services.chunking import structure_aware_chunk
    from app.services.embedding import embed_texts

    # ... (keep existing element query)

    # Convert elements to dicts for structure_aware_chunk
    element_dicts = [
        {
            "element_type": elem.element_type,
            "content_text": elem.content_text,
            "page_number": elem.page_number,
            "section_path": getattr(elem, "section_path", None),
            "element_uid": str(elem.element_uid) if elem.element_uid else "",
            "element_order": elem.element_order,
        }
        for elem in elements
        if elem.content_text
    ]

    structured_chunks = structure_aware_chunk(element_dicts)

    all_texts = []
    all_chunk_meta = []
    _seen_chunk_texts: set[str] = set()

    for sc in structured_chunks:
        if sc.text in _seen_chunk_texts:
            continue
        _seen_chunk_texts.add(sc.text)
        all_texts.append(sc.text)
        all_chunk_meta.append(sc)
```

Then update the loop that creates `chunk_values` to use `StructuredChunk` metadata (page_number, modality, etc.) instead of the raw element.

**Step 3: Run tests**

Run: `pytest tests/unit/test_chunking.py tests/unit/test_pipeline.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat: wire structure-aware chunker into ingest pipeline"
```

---

### Task 10: Add minimum score threshold and image oversample

**Files:**
- Modify: `app/api/v1/retrieval.py` — `_text_vector_search`, `_image_vector_search`
- Modify: `app/services/qdrant_store.py` — add `score_threshold` param
- Test: `tests/unit/test_retrieval_helpers.py`

**Step 1: Add score threshold to Qdrant search**

In `app/services/qdrant_store.py`, add `score_threshold` parameter to `search_text_vectors_async` and `search_image_vectors_async`.

**Step 2: Filter by threshold in `_text_vector_search`**

After getting hits from Qdrant:
```python
    # Drop results below minimum score threshold
    min_score = _settings.retrieval_min_score_threshold
    hits = [h for h in hits if h.get("score", 0.0) >= min_score]
```

**Step 3: Add oversample to `_image_vector_search`**

In `_image_vector_search` (line 347), replace:
```python
    limit=body.top_k,
```
with:
```python
    limit=min(
        body.top_k * _settings.retrieval_diversity_oversample_factor,
        _settings.retrieval_diversity_max_candidates,
    ),
```

And add diversification + trim after building results.

**Step 4: Run tests**

Run: `pytest tests/unit/test_retrieval_helpers.py tests/unit/test_retrieval_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/v1/retrieval.py app/services/qdrant_store.py tests/
git commit -m "feat: add min score threshold, image search oversample"
```

---

### Task 11: Fix GraphRAG global and local search

**Files:**
- Modify: `app/services/graphrag_service.py:135-179` — `global_search`
- Modify: `app/api/v1/retrieval.py` — use BM25 scores for local
- Test: `tests/unit/test_graphrag_service.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_service.py`:

```python
def test_global_search_filters_by_query(mock_db_session):
    """global_search should filter community reports by query relevance."""
    from app.services.graphrag_service import global_search
    # Verify the SQL includes a WHERE clause referencing the query
    import inspect
    source = inspect.getsource(global_search)
    assert "WHERE" in source or "where" in source, "global_search must filter by query text"
```

**Step 2: Fix `global_search`**

Replace the SQL in `global_search` (line 150-158) with fulltext filtering:

```python
        stmt = text("""
            SELECT cr.report_text, cr.summary, cr.rank,
                   gc.community_id, gc.title, gc.level,
                   ts_rank_cd(
                       to_tsvector('english', cr.report_text),
                       plainto_tsquery('english', :query)
                   ) AS relevance
            FROM retrieval.graphrag_community_reports cr
            JOIN retrieval.graphrag_communities gc
                ON cr.community_id = gc.community_id
            WHERE to_tsvector('english', cr.report_text) @@ plainto_tsquery('english', :query)
            ORDER BY relevance DESC, cr.rank DESC NULLS LAST
            LIMIT :limit
        """)
        result = db_session.execute(stmt, {"query": query, "limit": limit})
```

**Step 3: Fix local search scoring**

In `app/api/v1/retrieval.py`, where GraphRAG local results are scored (find the `max(0.5, 1.0 - (i * 0.05))` pattern), replace with the actual BM25 score from `fulltext_search_entity`:

```python
    # Use the fulltext score from Neo4j instead of rank-based scoring
    score = match.get("node", {}).get("score", 0.5)
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_graphrag_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/graphrag_service.py app/api/v1/retrieval.py tests/unit/test_graphrag_service.py
git commit -m "fix: graphrag global filters by query, local uses BM25 scores"
```

---

### Task 12: Fix canonicalization fuzzy threshold

**Files:**
- Modify: `app/services/canonicalization.py:138-148`
- Modify: `app/services/neo4j_graph.py` — update fulltext index
- Test: `tests/unit/test_canonicalization.py`

**Step 1: Normalize scores in `_fuzzy_match`**

Update `_fuzzy_match` in `app/services/canonicalization.py` (line 138):

```python
def _fuzzy_match(driver, name: str, entity_type: str) -> Optional[str]:
    """Use fulltext search for fuzzy matching with normalized scores."""
    from app.services.neo4j_graph import fulltext_search_entity

    results = fulltext_search_entity(driver, name, limit=5)
    if not results:
        return None

    # Normalize Lucene BM25 scores to 0-1 range using the max score
    max_score = max(r["score"] for r in results) if results else 1.0
    for r in results:
        if r["entity_type"] == entity_type:
            normalized_score = r["score"] / max_score if max_score > 0 else 0
            if normalized_score > FUZZY_THRESHOLD:
                candidate = r.get("canonical_name") or r["name"]
                if candidate != name:
                    return candidate
    return None
```

**Step 2: Update fulltext index to include canonical_name**

In `app/services/neo4j_graph.py`, find the fulltext index creation and add `canonical_name`:

```cypher
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.canonical_name]
```

**Step 3: Run tests**

Run: `pytest tests/unit/test_canonicalization.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add app/services/canonicalization.py app/services/neo4j_graph.py tests/unit/test_canonicalization.py
git commit -m "fix: normalize fuzzy match scores, include canonical_name in fulltext index"
```

---

### Task 13: Add model download to manage.sh

**Files:**
- Modify: `manage.sh`

**Step 1: Add model download step to `cmd_start`**

Add after `dc "${profile_args[@]}" up -d --build` (line 143) and before health checks:

```bash
  # Pre-download ML models (air-gapped: must be available before first query)
  header "Pre-downloading ML models"

  # Download sentence-transformers models inside the API container
  info "Downloading text embedding model (BGE-large-en-v1.5)..."
  dc exec -T api python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-large-en-v1.5')
print('BGE model ready')
" 2>/dev/null && info "BGE model ready" || warn "BGE model download failed (will retry on first use)"

  info "Downloading reranker model (bge-reranker-v2-m3)..."
  dc exec -T api python -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-v2-m3')
print('Reranker model ready')
" 2>/dev/null && info "Reranker model ready" || warn "Reranker model download failed (will retry on first use)"

  # Pull Ollama model via the external Ollama service
  if command -v ollama &>/dev/null; then
    info "Pulling Ollama model (llama3.1:8b)..."
    ollama pull llama3.1:8b 2>/dev/null && info "Ollama model ready" || warn "Ollama pull failed"
  else
    info "Ollama not found locally — ensure llama3.1:8b is available on the Ollama server"
  fi
```

**Step 2: Test manually**

Run: `./manage.sh --start` (or inspect the script for correctness)

**Step 3: Commit**

```bash
git add manage.sh
git commit -m "feat: pre-download reranker and LLM models during manage.sh --start"
```

---

### Task 14: Re-score expanded chunks independently

**Files:**
- Modify: `app/api/v1/retrieval.py` — `_expand_via_ontology` section

**Step 1: Update ontology expansion to compute independent scores**

In the `_expand_via_ontology` section of `_multi_modal_pipeline`, after fetching expanded chunks from Neo4j, embed their text and compute cosine similarity against the query:

```python
    # Re-score expanded chunks with their own embedding similarity
    from app.services.embedding import embed_texts
    import numpy as np

    if expanded_chunks and body.query_text:
        query_emb = np.array(embed_texts([body.query_text], query=True)[0])
        for chunk in expanded_chunks:
            if chunk.content_text:
                chunk_emb = np.array(embed_texts([chunk.content_text])[0])
                cos_sim = float(np.dot(query_emb, chunk_emb))
                chunk.score = compute_fusion_score(
                    semantic_score=cos_sim,
                    ontology_rel_type=chunk._ontology_rel_type,
                    ontology_hops=chunk._ontology_hops,
                )
```

Note: This adds latency for expanded chunks. The embed call should reuse the cached model. Consider batching all expanded chunk texts into a single `embed_texts` call.

**Step 2: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "feat: re-score ontology-expanded chunks with independent embeddings"
```

---

### Task 15: Sync base.yaml with base_v1.yaml

**Files:**
- Modify: `ontology/base.yaml` — ensure it matches `base_v1.yaml` or is a symlink

**Step 1: Check current state**

```bash
diff ontology/base.yaml ontology/base_v1.yaml
```

**Step 2: Replace base.yaml with symlink**

```bash
cd ontology && ln -sf base_v1.yaml base.yaml
```

Or copy the file if symlinks are problematic in Docker volumes.

**Step 3: Commit**

```bash
git add ontology/base.yaml
git commit -m "fix: sync base.yaml with base_v1.yaml"
```

---

### Task 16: Run full test suite and update README

**Files:**
- Run: `./scripts/run_tests.sh`
- Modify: `README.md` — update with new features

**Step 1: Run full test suite**

Run: `./manage.sh --test`

Fix any failures discovered during the run.

**Step 2: Update README.md**

Add reranker to the technology stack table. Update the retrieval modes section to mention cross-encoder reranking. Note the upgraded default LLM model.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with quality pass changes"
```
