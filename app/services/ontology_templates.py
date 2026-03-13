"""Convert ontology YAML entity/relationship types into Pydantic extraction templates.

These templates are used by the LLM-powered graph extraction pipeline
(docling-graph or direct LLM prompting) to produce typed entities and
relationships from document text.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "base.yaml"

# Python type mapping from YAML schema types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load and return the ontology YAML as a dict."""
    p = path or _ONTOLOGY_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def generate_entity_templates(
    ontology: dict[str, Any],
) -> dict[str, type[BaseModel]]:
    """Dynamically create a Pydantic model for each entity type in the ontology.

    Returns a dict mapping entity type name (e.g. "EQUIPMENT_SYSTEM") to its
    generated Pydantic class.
    """
    templates: dict[str, type[BaseModel]] = {}

    for entity_def in ontology.get("entity_types", []):
        name = entity_def["name"]
        description = entity_def.get("description", "")
        props_schema = entity_def.get("properties", {}).get("properties", {})

        # Build field definitions for create_model
        fields: dict[str, Any] = {}
        for prop_name, prop_def in props_schema.items():
            py_type = _TYPE_MAP.get(prop_def.get("type", "string"), str)
            field_desc = prop_def.get("description", "")
            enum_vals = prop_def.get("enum")

            if enum_vals:
                field_desc += f" (one of: {', '.join(str(v) for v in enum_vals)})"

            fields[prop_name] = (
                Optional[py_type],
                Field(default=None, description=field_desc or None),
            )

        model = create_model(
            name,
            __base__=BaseModel,
            __doc__=description,
            **fields,
        )
        # Attach metadata for graph extraction
        model.model_config["json_schema_extra"] = {
            "is_entity": True,
            "entity_type": name,
            "graph_id_fields": ["name"],
        }
        templates[name] = model

    return templates


def generate_relationship_templates(
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return relationship type definitions as structured dicts.

    Each dict contains: name, description, source_type, target_type, cardinality.
    """
    relationships = []
    for rel_def in ontology.get("relationship_types", []):
        relationships.append({
            "name": rel_def["name"],
            "description": rel_def.get("description", ""),
            "source_type": rel_def.get("source_type"),
            "target_type": rel_def.get("target_type"),
            "cardinality": rel_def.get("cardinality", "many_to_many"),
        })
    return relationships


class ExtractedEntity(BaseModel):
    """An entity extracted from document text."""
    entity_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelationship(BaseModel):
    """A relationship extracted from document text."""
    relationship_type: str
    from_name: str
    from_type: str
    to_name: str
    to_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DocumentExtractionResult(BaseModel):
    """Complete extraction result for a document or text chunk."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def build_extraction_prompt(
    ontology: dict[str, Any],
    text: str,
    max_text_length: int = 8000,
    *,
    compact_ontology: bool = False,
    max_properties_per_entity: int = 8,
    few_shot: bool = False,
) -> str:
    """Build an LLM prompt for entity/relationship extraction using the ontology.

    This prompt instructs the LLM to return a JSON object matching the
    DocumentExtractionResult schema.
    """
    # Truncate text if needed
    if len(text) > max_text_length:
        text = text[:max_text_length] + "\n[...truncated...]"

    entity_descriptions = []
    for et in ontology.get("entity_types", []):
        props = list(et.get("properties", {}).get("properties", {}).keys())
        if compact_ontology:
            prop_preview = ", ".join(props[:max_properties_per_entity]) or "none"
            if len(props) > max_properties_per_entity:
                prop_preview += ", ..."
            entity_descriptions.append(f"  - {et['name']} (props: {prop_preview})")
        else:
            prop_details = []
            for pname, pdef in et.get("properties", {}).get("properties", {}).items():
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

    rel_descriptions = []
    for rt in ontology.get("relationship_types", []):
        src = rt.get("source_type") or "any"
        tgt = rt.get("target_type") or "any"
        if compact_ontology:
            rel_descriptions.append(f"  - {rt['name']} ({src} -> {tgt})")
        else:
            rel_descriptions.append(
                f"  - {rt['name']}: {rt.get('description', '')} ({src} -> {tgt})"
            )

    few_shot_block = ""
    if few_shot:
        few_shot_block = '''
## Example
Input: "The Patriot PAC-3 missile system uses MIL-STD-1553B for internal data bus communication. The guidance section contains a Ka-band seeker operating at 35 GHz."
Output: {"entities": [{"entity_type": "EQUIPMENT_SYSTEM", "name": "Patriot PAC-3", "properties": {"name": "Patriot PAC-3"}, "confidence": 0.95}, {"entity_type": "STANDARD", "name": "MIL-STD-1553B", "properties": {"designation": "MIL-STD-1553B"}, "confidence": 0.95}, {"entity_type": "SUBSYSTEM", "name": "Guidance Section", "properties": {"name": "Guidance Section", "function": "Terminal phase target tracking and guidance"}, "confidence": 0.85}, {"entity_type": "COMPONENT", "name": "Ka-band Seeker", "properties": {"name": "Ka-band Seeker"}, "confidence": 0.80}], "relationships": [{"relationship_type": "MEETS_STANDARD", "from_name": "Patriot PAC-3", "from_type": "EQUIPMENT_SYSTEM", "to_name": "MIL-STD-1553B", "to_type": "STANDARD", "properties": {}, "confidence": 0.90}, {"relationship_type": "IS_SUBSYSTEM_OF", "from_name": "Guidance Section", "from_type": "SUBSYSTEM", "to_name": "Patriot PAC-3", "to_type": "EQUIPMENT_SYSTEM", "properties": {}, "confidence": 0.85}, {"relationship_type": "CONTAINS", "from_name": "Guidance Section", "from_type": "SUBSYSTEM", "to_name": "Ka-band Seeker", "to_type": "COMPONENT", "properties": {}, "confidence": 0.80}]}
'''

    prompt = f"""Extract entities and relationships from the following military/defense document text.

## Entity Types
{chr(10).join(entity_descriptions)}

## Relationship Types
{chr(10).join(rel_descriptions)}
{few_shot_block}
## Instructions
1. Identify all entities mentioned in the text that match the entity types above.
2. For each entity, extract its name and any available properties.
3. Identify relationships between entities.
4. Return a JSON object matching this schema:

```json
{{
  "entities": [
    {{
      "entity_type": "string (one of the entity types listed above)",
      "name": "string (canonical name of the entity)",
      "properties": {{"string": "any"}},
      "confidence": "number (0.0 to 1.0)"
    }}
  ],
  "relationships": [
    {{
      "relationship_type": "string (one of the relationship types listed above)",
      "from_name": "string (source entity name)",
      "from_type": "string (source entity type)",
      "to_name": "string (target entity name)",
      "to_type": "string (target entity type)",
      "properties": {{"string": "any"}},
      "confidence": "number (0.0 to 1.0)"
    }}
  ]
}}
```

Only extract entities and relationships you are confident about. Set confidence accordingly.
Only use entity types and relationship types from the lists above. Do not invent new types.
Place each extracted value in the correct property field based on its description.
Return ONLY valid JSON.

## Document Text
{text}
"""
    return prompt


def load_validation_matrix(
    path: Path | None = None,
) -> set[tuple[str, str, str]]:
    """Load the ontology validation matrix as a set of (source, rel, target) triples.

    Returns an empty set if the ontology doesn't define a validation_matrix.
    """
    ontology = load_ontology(path)
    matrix: set[tuple[str, str, str]] = set()
    for entry in ontology.get("validation_matrix", []):
        source = entry.get("source", "") or entry.get("source_type", "")
        rel = entry.get("relationship", "")
        target = entry.get("target", "") or entry.get("target_type", "")
        if source and rel and target:
            matrix.add((source, rel, target))
    return matrix


def build_entity_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all entity type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return [et["name"] for et in ontology.get("entity_types", [])]


def build_relationship_type_names(ontology: dict[str, Any] | None = None) -> list[str]:
    """Return a list of all relationship type names from the ontology."""
    if ontology is None:
        ontology = load_ontology()
    return [rt["name"] for rt in ontology.get("relationship_types", [])]
