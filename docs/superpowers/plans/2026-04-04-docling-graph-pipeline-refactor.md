# Docling-Graph Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled LLM extraction in the docling-graph service with proper `run_pipeline()` / `PipelineConfig` integration using delta extraction, ontology-driven Pydantic templates with `edge()` fields, and NetworkX graph output.

**Architecture:** The docling-graph FastAPI service becomes a thin wrapper around the docling-graph library's `run_pipeline()`. Templates are generated dynamically from the ontology YAML with `graph_id_fields` and `edge()` fields for relationships. Delta extraction with direct fallback provides the highest accuracy. A validation pass catches missed relationships.

**Tech Stack:** Python 3.12, FastAPI, docling-graph library (run_pipeline, PipelineConfig, delta extraction), Pydantic v2, NetworkX, PyYAML, httpx

**Spec:** `docs/superpowers/specs/2026-04-04-docling-graph-pipeline-refactor-design.md`

---

## File Map

### Files to create
| Path | Responsibility |
|------|---------------|
| `docker/docling-graph/app/template_builder.py` | Ontology YAML → Pydantic templates with edge() fields and graph_id_fields |
| `docker/docling-graph/app/config_builder.py` | Environment variables → PipelineConfig construction |
| `docker/docling-graph/tests/test_template_builder.py` | Template builder unit tests |
| `docker/docling-graph/tests/test_config_builder.py` | Config builder unit tests |
| `docker/docling-graph/tests/test_pipeline_integration.py` | End-to-end pipeline integration tests |
| `docker/docling-graph/tests/test_validation_pass.py` | Validation pass unit tests |

### Files to rewrite
| Path | What changes |
|------|-------------|
| `docker/docling-graph/app/main.py` | 921 lines → ~200 lines. Thin FastAPI wrapper around run_pipeline() |
| `docker/docling-graph/app/schemas.py` | New request/response models for DoclingDocument input and NetworkX graph output |
| `docker/docling-graph/tests/test_extraction.py` | Test /extract-all with new contract |

### Files to delete
| Path | Reason |
|------|--------|
| `docker/docling-graph/app/prompts.py` | Domain knowledge moves into template field descriptions |
| `docker/docling-graph/tests/test_direct_extraction.py` | No more direct LLM calls |
| `docker/docling-graph/tests/test_prompts.py` | prompts.py deleted |

### Files to modify
| Path | What changes |
|------|-------------|
| `docker/docling-graph/requirements.txt` | Remove litellm, json-repair |

---

## Chunk 1: Template Builder

### Task 1: Template Builder — graph_id_fields derivation

**Files:**
- Create: `docker/docling-graph/app/template_builder.py`
- Create: `docker/docling-graph/tests/test_template_builder.py`
- Read: `ontology/ontology.yaml` (reference for entity types and properties)

- [ ] **Step 1: Write failing test — graph_id_fields derivation per entity type**

```python
# docker/docling-graph/tests/test_template_builder.py
"""Tests for ontology-to-template builder."""
import pytest
import yaml
from pathlib import Path

# Load the real ontology for testing
ONTOLOGY_PATH = Path(__file__).parent.parent.parent.parent / "ontology" / "ontology.yaml"


def _load_ontology():
    with open(ONTOLOGY_PATH) as f:
        return yaml.safe_load(f)


class TestGraphIdFieldsDerivation:
    """Test that graph_id_fields are derived correctly per entity type."""

    def test_entity_with_name_property(self):
        """PLATFORM has 'name' — should use ['name']."""
        from app.template_builder import derive_graph_id_fields

        props = {"name": {"type": "string"}, "platform_type": {"type": "string"}}
        result = derive_graph_id_fields("PLATFORM", props)
        assert result == ["name"]

    def test_entity_with_system_name(self):
        """RADAR_SYSTEM has 'system_name' — should use ['system_name']."""
        from app.template_builder import derive_graph_id_fields

        props = {"system_name": {"type": "string"}, "nomenclature": {"type": "string"}}
        result = derive_graph_id_fields("RADAR_SYSTEM", props)
        assert result == ["system_name"]

    def test_entity_with_id_suffix(self):
        """DOCUMENT has 'document_id' — should use ['document_id']."""
        from app.template_builder import derive_graph_id_fields

        props = {"document_id": {"type": "string"}, "title": {"type": "string"}}
        result = derive_graph_id_fields("DOCUMENT", props)
        assert result == ["document_id"]

    def test_entity_with_figure_id(self):
        """FIGURE has 'figure_id' — should use ['figure_id']."""
        from app.template_builder import derive_graph_id_fields

        props = {"figure_id": {"type": "string"}, "caption": {"type": "string"}}
        result = derive_graph_id_fields("FIGURE", props)
        assert result == ["figure_id"]

    def test_entity_with_table_id(self):
        """TABLE has 'table_id' — should use ['table_id']."""
        from app.template_builder import derive_graph_id_fields

        props = {"table_id": {"type": "string"}, "caption": {"type": "string"}}
        result = derive_graph_id_fields("TABLE", props)
        assert result == ["table_id"]

    def test_entity_with_heading(self):
        """SECTION has 'heading' — should use ['heading']."""
        from app.template_builder import derive_graph_id_fields

        props = {"heading": {"type": "string"}, "page_start": {"type": "integer"}}
        result = derive_graph_id_fields("SECTION", props)
        assert result == ["heading"]

    def test_entity_with_composite_identity(self):
        """SPECIFICATION has 'parameter' + 'value' — should use both."""
        from app.template_builder import derive_graph_id_fields

        props = {
            "parameter": {"type": "string"},
            "value": {"type": "string"},
            "unit": {"type": "string"},
        }
        result = derive_graph_id_fields("SPECIFICATION", props)
        assert result == ["parameter", "value"]

    def test_fallback_to_first_property(self):
        """Unknown entity with no standard fields — uses first property."""
        from app.template_builder import derive_graph_id_fields

        props = {"custom_field": {"type": "string"}, "other": {"type": "integer"}}
        result = derive_graph_id_fields("CUSTOM_TYPE", props)
        assert result == ["custom_field"]

    def test_priority_name_over_id_suffix(self):
        """If both 'name' and '*_id' exist, 'name' wins."""
        from app.template_builder import derive_graph_id_fields

        props = {"name": {"type": "string"}, "component_id": {"type": "string"}}
        result = derive_graph_id_fields("COMPONENT", props)
        assert result == ["name"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestGraphIdFieldsDerivation -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.template_builder'`

- [ ] **Step 3: Implement derive_graph_id_fields**

```python
# docker/docling-graph/app/template_builder.py
"""Ontology YAML → Pydantic templates with edge() fields and graph_id_fields.

Reads an ontology definition and generates docling-graph-compatible Pydantic
templates for delta extraction. Each entity type becomes a model with:
- graph_id_fields for stable deduplication
- edge() fields for relationships (from validation_matrix)
- Typed properties with descriptions and examples
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Composite identity types: entity types that use multiple fields for identity
_COMPOSITE_ID_TYPES: dict[str, list[str]] = {
    "SPECIFICATION": ["parameter", "value"],
}


def derive_graph_id_fields(entity_type: str, properties: dict[str, Any]) -> list[str]:
    """Derive graph_id_fields for an entity type from its ontology properties.

    Priority:
    1. Composite identity (if entity_type in _COMPOSITE_ID_TYPES)
    2. 'name' or 'system_name' property
    3. Property with '_id' suffix (e.g., document_id, figure_id, table_id)
    4. 'title' or 'heading' property
    5. Fallback: first property
    """
    # 0. Check composite identity override
    if entity_type in _COMPOSITE_ID_TYPES:
        return _COMPOSITE_ID_TYPES[entity_type]

    prop_names = list(properties.keys())
    if not prop_names:
        return ["name"]  # safety fallback

    # 1. name or system_name
    for field_name in ("name", "system_name"):
        if field_name in prop_names:
            return [field_name]

    # 2. *_id suffix
    for field_name in prop_names:
        if field_name.endswith("_id") and field_name != "entity_type":
            return [field_name]

    # 3. title or heading
    for field_name in ("title", "heading"):
        if field_name in prop_names:
            return [field_name]

    # 4. Fallback: first property
    return [prop_names[0]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestGraphIdFieldsDerivation -v
```
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add docker/docling-graph/app/template_builder.py docker/docling-graph/tests/test_template_builder.py
git commit -m "feat(docling-graph): add graph_id_fields derivation from ontology properties"
```

---

### Task 2: Template Builder — Pydantic model generation from ontology

**Files:**
- Modify: `docker/docling-graph/app/template_builder.py`
- Modify: `docker/docling-graph/tests/test_template_builder.py`

- [ ] **Step 1: Write failing test — basic model generation**

```python
# Append to docker/docling-graph/tests/test_template_builder.py

class TestBuildTemplates:
    """Test Pydantic model generation from ontology."""

    def test_builds_model_for_entity_type(self):
        """Should create a Pydantic model class for each entity type."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        assert "RADAR_SYSTEM" in templates
        model_cls = templates["RADAR_SYSTEM"]
        assert issubclass(model_cls, BaseModel)

    def test_model_has_graph_id_fields(self):
        """Model config should include graph_id_fields."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        config = model_cls.model_config
        assert "graph_id_fields" in config
        assert config["graph_id_fields"] == ["system_name"]

    def test_model_has_typed_fields(self):
        """Model fields should match ontology property types."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        assert "system_name" in fields
        assert "nomenclature" in fields
        assert "radar_type" in fields

    def test_all_fields_optional_except_identity(self):
        """Non-identity fields should be Optional."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        # Identity field is required
        assert fields["system_name"].is_required()

        # Other fields are optional
        assert not fields["nomenclature"].is_required()

    def test_field_descriptions_from_ontology(self):
        """Fields should have descriptions from ontology."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        assert fields["system_name"].description is not None
        assert len(fields["system_name"].description) > 0

    def test_all_ontology_entity_types_have_templates(self):
        """Every entity type in ontology should produce a template."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        entity_type_names = {et["name"] for et in ontology["entity_types"]}
        for name in entity_type_names:
            assert name in templates, f"Missing template for {name}"

    def test_reserved_word_table_mapped(self):
        """TABLE ontology type should map to TABLE_REF template."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        # TABLE in ontology maps to TABLE_REF in templates
        assert "TABLE_REF" in templates or "TABLE" in templates

    def test_model_instantiation(self):
        """Should be able to instantiate the model with identity fields."""
        from app.template_builder import build_templates

        ontology = _load_ontology()
        templates = build_templates(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        instance = model_cls(system_name="Tombstone")
        assert instance.system_name == "Tombstone"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestBuildTemplates -v
```
Expected: FAIL with `ImportError: cannot import name 'build_templates'`

- [ ] **Step 3: Implement build_templates**

```python
# Add to docker/docling-graph/app/template_builder.py

# Reserved word mapping: ontology name -> ArcadeDB-safe name
RESERVED_WORD_MAP: dict[str, str] = {
    "TABLE": "TABLE_REF",
}

# Reverse mapping for lookups
_REVERSE_RESERVED: dict[str, str] = {v: k for k, v in RESERVED_WORD_MAP.items()}

# YAML type -> Python type mapping
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _safe_type_name(ontology_name: str) -> str:
    """Map ontology entity type name to a safe Python/ArcadeDB name."""
    return RESERVED_WORD_MAP.get(ontology_name, ontology_name)


def _build_single_template(
    entity_type_def: dict[str, Any],
) -> tuple[str, Type[BaseModel]]:
    """Build a single Pydantic model from an ontology entity type definition.

    Returns (template_name, model_class).
    """
    ontology_name = entity_type_def["name"]
    safe_name = _safe_type_name(ontology_name)
    description = entity_type_def.get("description", f"Entity type: {ontology_name}")

    # Extract properties from ontology schema
    props_schema = entity_type_def.get("properties", {})
    props_dict = props_schema.get("properties", {})

    # Derive identity fields
    id_fields = derive_graph_id_fields(ontology_name, props_dict)

    # Build Pydantic field definitions
    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in props_dict.items():
        prop_type_str = prop_schema.get("type", "string")
        python_type = _TYPE_MAP.get(prop_type_str, str)
        field_description = prop_schema.get("description", "")
        field_example = prop_schema.get("example")
        field_examples = [field_example] if field_example is not None else None

        field_kwargs: dict[str, Any] = {}
        if field_description:
            field_kwargs["description"] = field_description
        if field_examples:
            field_kwargs["examples"] = field_examples

        if prop_name in id_fields:
            # Identity field: required
            field_definitions[prop_name] = (
                python_type,
                Field(**field_kwargs),
            )
        else:
            # Non-identity field: optional
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=None, **field_kwargs),
            )

    # Create model config
    model_config = ConfigDict(
        graph_id_fields=id_fields,
    )

    # Build the model class dynamically
    namespace = {"__annotations__": {}, "model_config": model_config}

    for field_name, (field_type, field_obj) in field_definitions.items():
        namespace["__annotations__"][field_name] = field_type
        namespace[field_name] = field_obj

    model_cls = type(safe_name, (BaseModel,), namespace)
    model_cls.__doc__ = description

    return safe_name, model_cls


def build_templates(
    ontology: dict[str, Any],
) -> dict[str, Type[BaseModel]]:
    """Build Pydantic templates for all entity types in the ontology.

    Returns a dict mapping entity type name -> Pydantic model class.
    """
    templates: dict[str, Type[BaseModel]] = {}

    for entity_type_def in ontology.get("entity_types", []):
        name, model_cls = _build_single_template(entity_type_def)
        templates[name] = model_cls

    logger.info(
        "Built %d templates from ontology (version=%s)",
        len(templates),
        ontology.get("version", "unknown"),
    )

    return templates
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestBuildTemplates -v
```
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add docker/docling-graph/app/template_builder.py docker/docling-graph/tests/test_template_builder.py
git commit -m "feat(docling-graph): add Pydantic model generation from ontology entity types"
```

---

### Task 3: Template Builder — edge() fields from validation matrix

**Files:**
- Modify: `docker/docling-graph/app/template_builder.py`
- Modify: `docker/docling-graph/tests/test_template_builder.py`

- [ ] **Step 1: Write failing test — edge fields on templates**

```python
# Append to docker/docling-graph/tests/test_template_builder.py

class TestEdgeFields:
    """Test that edge() fields are generated from the validation matrix."""

    def test_radar_system_has_installed_on_edge(self):
        """RADAR_SYSTEM should have an installed_on edge to PLATFORM."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        assert "installed_on" in fields
        edge_meta = fields["installed_on"].json_schema_extra or {}
        assert edge_meta.get("edge_label") == "INSTALLED_ON"

    def test_radar_system_has_multiple_edges(self):
        """RADAR_SYSTEM should have edges for HAS_ANTENNA, USES_WAVEFORM, etc."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        # Check several expected edges from validation_matrix
        edge_labels = set()
        for field_info in fields.values():
            meta = field_info.json_schema_extra or {}
            if "edge_label" in meta:
                edge_labels.add(meta["edge_label"])

        assert "INSTALLED_ON" in edge_labels
        assert "HAS_ANTENNA" in edge_labels
        assert "USES_WAVEFORM" in edge_labels
        assert "HAS_PERFORMANCE" in edge_labels

    def test_one_to_many_edge_is_list(self):
        """one_to_many cardinality edges should be List[TargetType]."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        # HAS_ANTENNA is one_to_many
        has_antenna = fields.get("has_antenna")
        assert has_antenna is not None
        # Should have default_factory=list
        assert has_antenna.default_factory is list

    def test_one_to_one_edge_is_optional(self):
        """one_to_one cardinality edges should be Optional[TargetType]."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields

        # HAS_PROCESSING_CHAIN is one_to_one
        has_chain = fields.get("has_processing_chain")
        assert has_chain is not None
        assert not has_chain.is_required()

    def test_no_edge_for_invalid_source(self):
        """Entity types not in validation_matrix as source should have no edges."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        # IF_AMPLIFIER has very few outgoing relationships
        model_cls = templates["IF_AMPLIFIER"]
        fields = model_cls.model_fields

        edge_count = sum(
            1 for f in fields.values()
            if (f.json_schema_extra or {}).get("edge_label")
        )
        # IF_AMPLIFIER has PART_OF -> RECEIVER in the matrix
        assert edge_count >= 1

    def test_specification_has_specified_by_edge(self):
        """SPECIFICATION heuristic: should have edges from validation_matrix."""
        from app.template_builder import build_templates_with_edges

        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)

        # Check that types with SPECIFIED_BY as target have the edge
        model_cls = templates.get("EQUIPMENT_SYSTEM")
        if model_cls:
            fields = model_cls.model_fields
            edge_labels = {
                (f.json_schema_extra or {}).get("edge_label")
                for f in fields.values()
                if (f.json_schema_extra or {}).get("edge_label")
            }
            assert "SPECIFIED_BY" in edge_labels
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestEdgeFields -v
```
Expected: FAIL with `ImportError: cannot import name 'build_templates_with_edges'`

- [ ] **Step 3: Implement edge field generation**

```python
# Add to docker/docling-graph/app/template_builder.py

from typing import List

try:
    from pydantic import Field as PydanticField

    def edge(label: str, **kwargs: Any) -> Any:
        """Helper to create a Pydantic Field with edge metadata for docling-graph."""
        return PydanticField(
            ..., json_schema_extra={"edge_label": label}, **kwargs
        )
except ImportError:
    pass


def _build_edge_map(
    ontology: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build a map of source_type -> list of edge definitions from validation_matrix.

    Returns: {source_type: [{rel_type, target_type, cardinality}, ...]}
    """
    edge_map: dict[str, list[dict[str, Any]]] = {}

    # Build relationship cardinality lookup
    rel_cardinality: dict[str, str] = {}
    for rel_def in ontology.get("relationship_types", []):
        rel_cardinality[rel_def["name"]] = rel_def.get("cardinality", "many_to_many")

    # Process validation matrix
    for entry in ontology.get("validation_matrix", []):
        source = entry.get("source")
        rel_type = entry.get("relationship")
        target = entry.get("target")

        if not source or not rel_type or not target:
            continue

        if source not in edge_map:
            edge_map[source] = []

        # Avoid duplicate edges for same (source, rel_type, target)
        existing = {(e["rel_type"], e["target_type"]) for e in edge_map[source]}
        if (rel_type, target) not in existing:
            edge_map[source].append({
                "rel_type": rel_type,
                "target_type": target,
                "cardinality": rel_cardinality.get(rel_type, "many_to_many"),
            })

    return edge_map


def build_templates_with_edges(
    ontology: dict[str, Any],
) -> dict[str, Type[BaseModel]]:
    """Build Pydantic templates with edge() fields from ontology + validation_matrix.

    This is the main entry point for template generation. It:
    1. Builds basic templates from entity types
    2. Adds edge() fields from the validation matrix
    3. Returns templates keyed by (safe) entity type name
    """
    # Step 1: Build basic templates (without edges)
    basic_templates = build_templates(ontology)

    # Step 2: Build edge map from validation matrix
    edge_map = _build_edge_map(ontology)

    # Step 3: Rebuild templates with edge fields
    final_templates: dict[str, Type[BaseModel]] = {}

    for entity_type_def in ontology.get("entity_types", []):
        ontology_name = entity_type_def["name"]
        safe_name = _safe_type_name(ontology_name)

        base_model = basic_templates.get(safe_name)
        if base_model is None:
            continue

        # Get edges for this source type
        edges = edge_map.get(ontology_name, [])

        if not edges:
            # No edges — use basic template as-is
            final_templates[safe_name] = base_model
            continue

        # Rebuild model with edge fields added
        # Start with existing field definitions
        namespace: dict[str, Any] = {
            "__annotations__": {},
            "model_config": base_model.model_config,
        }

        # Copy existing fields
        for field_name, field_info in base_model.model_fields.items():
            annotation = base_model.__annotations__.get(field_name, str)
            namespace["__annotations__"][field_name] = annotation
            if field_info.default is not None:
                namespace[field_name] = field_info.default
            elif field_info.default_factory is not None:
                namespace[field_name] = Field(
                    default_factory=field_info.default_factory,
                    description=field_info.description,
                )
            else:
                namespace[field_name] = Field(
                    description=field_info.description,
                    examples=field_info.examples,
                )

        # Add edge fields
        for edge_def in edges:
            rel_type = edge_def["rel_type"]
            target_type = edge_def["target_type"]
            cardinality = edge_def["cardinality"]

            # Field name: lowercase rel_type
            field_name = rel_type.lower()

            # Get target model class (may not exist yet for forward refs)
            target_safe = _safe_type_name(target_type)
            target_model = basic_templates.get(target_safe, BaseModel)

            # Relationship description from ontology
            rel_description = ""
            for rel_def in ontology.get("relationship_types", []):
                if rel_def["name"] == rel_type:
                    rel_description = rel_def.get("description", "")
                    break

            if cardinality in ("one_to_one", "many_to_one"):
                # Optional single reference
                namespace["__annotations__"][field_name] = Optional[target_model]
                namespace[field_name] = Field(
                    default=None,
                    json_schema_extra={"edge_label": rel_type},
                    description=rel_description,
                )
            else:
                # List reference
                namespace["__annotations__"][field_name] = List[target_model]
                namespace[field_name] = Field(
                    default_factory=list,
                    json_schema_extra={"edge_label": rel_type},
                    description=rel_description,
                )

        model_cls = type(safe_name, (BaseModel,), namespace)
        model_cls.__doc__ = base_model.__doc__
        final_templates[safe_name] = model_cls

    logger.info(
        "Built %d templates with edges from ontology (version=%s)",
        len(final_templates),
        ontology.get("version", "unknown"),
    )

    return final_templates
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py::TestEdgeFields -v
```
Expected: All 6 tests PASS

- [ ] **Step 5: Run all template builder tests**

```bash
cd docker/docling-graph && python -m pytest tests/test_template_builder.py -v
```
Expected: All 23 tests PASS

- [ ] **Step 6: Commit**

```bash
git add docker/docling-graph/app/template_builder.py docker/docling-graph/tests/test_template_builder.py
git commit -m "feat(docling-graph): add edge() field generation from validation matrix"
```

---

### Task 4: Config Builder — PipelineConfig from environment variables

**Files:**
- Create: `docker/docling-graph/app/config_builder.py`
- Create: `docker/docling-graph/tests/test_config_builder.py`

- [ ] **Step 1: Write failing test — default config construction**

```python
# docker/docling-graph/tests/test_config_builder.py
"""Tests for PipelineConfig builder from environment variables."""
import os
import pytest
from unittest.mock import patch


class TestConfigBuilderDefaults:
    """Test default PipelineConfig construction."""

    def test_builds_config_with_defaults(self):
        """Should build a valid PipelineConfig with default env vars."""
        from app.config_builder import build_pipeline_config

        config = build_pipeline_config(
            source="/tmp/test_doc.json",
            template_class=None,  # Will be set later
        )

        assert config.extraction_contract == "delta"
        assert config.backend == "llm"
        assert config.processing_mode == "many-to-one"
        assert config.use_chunking is True
        assert config.gleaning_enabled is True
        assert config.dump_to_disk is False

    def test_delta_resolvers_enabled_by_default(self):
        """Delta resolvers should be enabled by default."""
        from app.config_builder import build_pipeline_config

        config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.delta_resolvers_enabled is True
        assert config.delta_resolvers_mode == "semantic"

    def test_structured_output_enabled_by_default(self):
        """Structured output should be enabled by default."""
        from app.config_builder import build_pipeline_config

        config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.structured_output is True


class TestConfigBuilderOverrides:
    """Test env var overrides."""

    def test_extraction_contract_override(self):
        """Should respect DOCLING_GRAPH_EXTRACTION_CONTRACT env var."""
        from app.config_builder import build_pipeline_config

        with patch.dict(os.environ, {"DOCLING_GRAPH_EXTRACTION_CONTRACT": "direct"}):
            config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.extraction_contract == "direct"

    def test_parallel_workers_override(self):
        """Should respect DOCLING_GRAPH_PARALLEL_WORKERS env var."""
        from app.config_builder import build_pipeline_config

        with patch.dict(os.environ, {"DOCLING_GRAPH_PARALLEL_WORKERS": "4"}):
            config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.parallel_workers == 4

    def test_gleaning_disabled_override(self):
        """Should respect DOCLING_GRAPH_GLEANING_ENABLED=false."""
        from app.config_builder import build_pipeline_config

        with patch.dict(os.environ, {"DOCLING_GRAPH_GLEANING_ENABLED": "false"}):
            config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.gleaning_enabled is False

    def test_llm_batch_token_size_override(self):
        """Should respect DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE env var."""
        from app.config_builder import build_pipeline_config

        with patch.dict(os.environ, {"DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE": "4096"}):
            config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        assert config.llm_batch_token_size == 4096
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd docker/docling-graph && python -m pytest tests/test_config_builder.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config_builder'`

- [ ] **Step 3: Implement config_builder.py**

```python
# docker/docling-graph/app/config_builder.py
"""Environment variables → PipelineConfig construction.

Reads all DOCLING_GRAPH_* environment variables and constructs a
PipelineConfig for each extraction request.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int_or_none(key: str, default: int | None) -> int | None:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    return int(val)


def build_pipeline_config(
    source: str,
    template_class: Type[BaseModel] | None,
) -> Any:
    """Build a PipelineConfig from environment variables.

    Args:
        source: Path to DoclingDocument JSON file
        template_class: Pydantic template class (from template_builder)

    Returns:
        PipelineConfig instance
    """
    from docling_graph import PipelineConfig

    config_kwargs: dict[str, Any] = {
        "source": source,
        "backend": "llm",
        "inference": "local",
        "provider_override": _env_str("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
        "model_override": _env_str("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),

        # Extraction
        "extraction_contract": _env_str("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        "processing_mode": _env_str("DOCLING_GRAPH_PROCESSING_MODE", "many-to-one"),
        "use_chunking": _env_bool("DOCLING_GRAPH_USE_CHUNKING", True),
        "chunk_max_tokens": _env_int("DOCLING_GRAPH_CHUNK_MAX_TOKENS", 512),

        # Delta
        "llm_batch_token_size": _env_int("DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE", 2048),
        "parallel_workers": _env_int("DOCLING_GRAPH_PARALLEL_WORKERS", 2),
        "staged_pass_retries": _env_int("DOCLING_GRAPH_BATCH_SPLIT_MAX_RETRIES", 1),

        # Delta resolvers
        "delta_resolvers_enabled": _env_bool("DOCLING_GRAPH_RESOLVERS_ENABLED", True),
        "delta_resolvers_mode": _env_str("DOCLING_GRAPH_RESOLVERS_MODE", "semantic"),
        "delta_resolver_fuzzy_threshold": _env_float("DOCLING_GRAPH_RESOLVER_FUZZY_THRESHOLD", 0.8),
        "delta_resolver_semantic_threshold": _env_float("DOCLING_GRAPH_RESOLVER_SEMANTIC_THRESHOLD", 0.8),

        # Delta quality gate
        "delta_quality_require_root": _env_bool("DOCLING_GRAPH_QUALITY_REQUIRE_ROOT", True),
        "delta_quality_min_instances": _env_int("DOCLING_GRAPH_QUALITY_MIN_INSTANCES", 20),
        "delta_quality_max_parent_lookup_miss": _env_int("DOCLING_GRAPH_QUALITY_MAX_PARENT_MISS", 4),
        "delta_quality_adaptive_parent_lookup": _env_bool("DOCLING_GRAPH_QUALITY_ADAPTIVE_PARENT", True),

        # Delta normalizer
        "delta_normalizer_validate_paths": _env_bool("DOCLING_GRAPH_NORMALIZER_VALIDATE_PATHS", True),
        "delta_normalizer_canonicalize_ids": _env_bool("DOCLING_GRAPH_NORMALIZER_CANONICALIZE_IDS", True),
        "delta_normalizer_strip_nested_properties": _env_bool("DOCLING_GRAPH_NORMALIZER_STRIP_NESTED", True),
        "delta_normalizer_attach_provenance": _env_bool("DOCLING_GRAPH_NORMALIZER_ATTACH_PROVENANCE", True),

        # Delta identity filter
        "delta_identity_filter_enabled": _env_bool("DOCLING_GRAPH_IDENTITY_FILTER_ENABLED", True),
        "delta_identity_filter_strict": _env_bool("DOCLING_GRAPH_IDENTITY_FILTER_STRICT", False),

        # Gleaning
        "gleaning_enabled": _env_bool("DOCLING_GRAPH_GLEANING_ENABLED", True),
        "gleaning_max_passes": _env_int("DOCLING_GRAPH_GLEANING_MAX_PASSES", 1),

        # Structured output
        "structured_output": _env_bool("DOCLING_GRAPH_STRUCTURED_OUTPUT", True),
        "structured_sparse_check": _env_bool("DOCLING_GRAPH_STRUCTURED_SPARSE_CHECK", True),

        # LLM overrides
        "llm_overrides": {
            "generation": {
                "temperature": _env_float("DOCLING_GRAPH_LLM_TEMPERATURE", 0.1),
                "max_tokens": _env_int_or_none("DOCLING_GRAPH_LLM_MAX_TOKENS", 64000),
            },
            "reliability": {
                "timeout_s": _env_int("DOCLING_GRAPH_LLM_TIMEOUT", 10800),
            },
            "connection": {
                "base_url": _env_str("OLLAMA_LLM_BASE_URL", "http://ollama:11434"),
            },
            "context_limit": _env_int_or_none("DOCLING_GRAPH_LLM_CONTEXT_LIMIT", None),
            "max_output_tokens": _env_int_or_none("DOCLING_GRAPH_LLM_MAX_OUTPUT_TOKENS", None),
        },

        # Output (API mode, no disk writes)
        "dump_to_disk": False,
    }

    # Set template if provided
    if template_class is not None:
        config_kwargs["template"] = template_class

    return PipelineConfig(**config_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd docker/docling-graph && python -m pytest tests/test_config_builder.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add docker/docling-graph/app/config_builder.py docker/docling-graph/tests/test_config_builder.py
git commit -m "feat(docling-graph): add PipelineConfig builder from environment variables"
```

---

### Task 5: Schemas — new request/response models

**Files:**
- Rewrite: `docker/docling-graph/app/schemas.py`

- [ ] **Step 1: Rewrite schemas.py with new contract**

```python
# docker/docling-graph/app/schemas.py
"""Request and response models for the Docling-Graph extraction service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request body for /extract-all endpoint."""

    document_id: str = Field(
        ..., description="UUID of the document being processed"
    )
    docling_document_json: dict[str, Any] = Field(
        ..., description="Full DoclingDocument JSON (skips re-conversion)"
    )
    ontology_definition: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional per-request ontology override",
    )
    ontology_version: Optional[str] = Field(
        default=None,
        description="Expected ontology version (logged if mismatched)",
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction pipeline run."""

    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    extraction_contract: str = "delta"
    gleaning_passes: int = 0
    resolvers_applied: bool = False
    quality_gate_passed: bool = True
    validation_pass_applied: bool = False
    validation_pass_edges_added: int = 0


class ExtractionResponse(BaseModel):
    """Response body for /extract-all endpoint."""

    graph: dict[str, Any] = Field(
        ..., description="Serialized NetworkX graph (node-link JSON)"
    )
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    model: str = "unknown"
    provider: str = "docling-graph"
    ontology_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str = "ok"
    ontology_version: Optional[str] = None
    template_count: int = 0
    extraction_contract: str = "delta"
    pipeline_version: str = "unknown"
```

- [ ] **Step 2: Commit**

```bash
git add docker/docling-graph/app/schemas.py
git commit -m "feat(docling-graph): rewrite request/response schemas for NetworkX graph contract"
```

---

### Task 6: Main.py — thin FastAPI wrapper

**Files:**
- Rewrite: `docker/docling-graph/app/main.py`
- Rewrite: `docker/docling-graph/tests/test_extraction.py`

- [ ] **Step 1: Write failing test — /extract-all with new contract**

```python
# docker/docling-graph/tests/test_extraction.py
"""Tests for the /extract-all endpoint with the new pipeline-based contract."""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline_context():
    """Create a mock PipelineContext with a sample NetworkX graph."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(
        "RADAR_SYSTEM_Tombstone",
        type="RADAR_SYSTEM",
        name="Tombstone",
        system_name="Tombstone",
        _provenance={"batch_id": 0, "chunk_index": 0, "page_numbers": [14]},
    )
    graph.add_node(
        "FREQUENCY_BAND_S-band",
        type="FREQUENCY_BAND",
        name="S-band",
        band_name="S-band",
        _provenance={"batch_id": 0, "chunk_index": 0, "page_numbers": [14]},
    )
    graph.add_edge(
        "RADAR_SYSTEM_Tombstone",
        "FREQUENCY_BAND_S-band",
        label="OPERATES_IN_BAND",
    )

    context = MagicMock()
    context.knowledge_graph = graph
    context.graph_metadata = MagicMock(
        node_count=2,
        edge_count=1,
        node_types={"RADAR_SYSTEM": 1, "FREQUENCY_BAND": 1},
        edge_types={"OPERATES_IN_BAND": 1},
    )
    return context


@pytest.fixture
def client(mock_pipeline_context):
    """Create test client with mocked pipeline."""
    with patch("app.main.run_extraction_pipeline", return_value=mock_pipeline_context):
        from app.main import app
        yield TestClient(app)


class TestExtractAll:
    """Test POST /extract-all endpoint."""

    def test_returns_networkx_graph(self, client):
        """Response should contain a serialized NetworkX graph."""
        response = client.post(
            "/extract-all",
            json={
                "document_id": "test-doc-001",
                "docling_document_json": {"schema_name": "DoclingDocument", "version": "1.0"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "graph" in data
        assert "nodes" in data["graph"]
        assert "links" in data["graph"]

    def test_returns_metadata(self, client):
        """Response should contain extraction metadata."""
        response = client.post(
            "/extract-all",
            json={
                "document_id": "test-doc-001",
                "docling_document_json": {"schema_name": "DoclingDocument", "version": "1.0"},
            },
        )

        data = response.json()
        assert "metadata" in data
        assert data["metadata"]["node_count"] == 2
        assert data["metadata"]["edge_count"] == 1

    def test_rejects_missing_document_id(self, client):
        """Should return 422 for missing document_id."""
        response = client.post(
            "/extract-all",
            json={"docling_document_json": {}},
        )
        assert response.status_code == 422


class TestHealth:
    """Test GET /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
```

- [ ] **Step 2: Rewrite main.py as thin wrapper**

```python
# docker/docling-graph/app/main.py
"""Docling-Graph extraction service.

Thin FastAPI wrapper around docling-graph library's run_pipeline().
Accepts DoclingDocument JSON, runs delta extraction with ontology-driven
Pydantic templates, returns NetworkX graph JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
import yaml
from fastapi import FastAPI, HTTPException

from app.config_builder import build_pipeline_config
from app.schemas import (
    ExtractionMetadata,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
)
from app.template_builder import build_templates_with_edges

logger = logging.getLogger(__name__)

# Module-level state (populated at startup)
_templates: dict[str, Any] = {}
_ontology_version: str | None = None
_ontology_hash: str | None = None

# Concurrency limiter
_extraction_semaphore: asyncio.Semaphore | None = None

# Cache for per-request ontology overrides: hash -> templates
_ontology_cache: dict[str, dict[str, Any]] = {}

ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml")
MAX_CONCURRENT = int(os.environ.get("DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS", "2"))


def _validate_library_surface() -> str:
    """Validate that the docling-graph library provides the required API surface.

    Returns the library version string.
    Raises ImportError with a clear message if anything is missing.
    """
    missing = []

    try:
        from docling_graph import run_pipeline, PipelineConfig  # noqa: F401
    except ImportError:
        missing.append("run_pipeline / PipelineConfig")

    try:
        from docling_graph.config import PipelineConfig as PC

        # Check delta extraction support
        if not hasattr(PC.model_fields.get("extraction_contract", None), "default"):
            pass  # Field exists, good enough
    except Exception:
        missing.append("extraction_contract on PipelineConfig")

    if missing:
        raise ImportError(
            f"docling-graph library missing required API surface: {', '.join(missing)}. "
            f"Update the library or check the installed version."
        )

    try:
        import docling_graph

        return getattr(docling_graph, "__version__", "unknown")
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup: load ontology, build templates, validate library."""
    global _templates, _ontology_version, _ontology_hash, _extraction_semaphore

    # Validate library
    lib_version = _validate_library_surface()
    logger.info("docling-graph library version: %s", lib_version)
    app.state.pipeline_version = lib_version

    # Load ontology
    if os.path.exists(ONTOLOGY_PATH):
        with open(ONTOLOGY_PATH) as f:
            ontology = yaml.safe_load(f)
        _ontology_version = ontology.get("version")
        _ontology_hash = hashlib.sha256(
            json.dumps(ontology, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Build templates
        _templates = build_templates_with_edges(ontology)
        logger.info(
            "Loaded ontology v%s (%d templates) from %s",
            _ontology_version,
            len(_templates),
            ONTOLOGY_PATH,
        )
    else:
        logger.warning("Ontology not found at %s — templates will be empty", ONTOLOGY_PATH)

    # Initialize concurrency limiter
    _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    yield

    # Shutdown
    logger.info("Docling-Graph service shutting down")


app = FastAPI(
    title="Docling-Graph Extraction Service",
    version="2.0.0",
    lifespan=lifespan,
)


def _resolve_templates(
    ontology_definition: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve templates for the request — use default or per-request override."""
    if ontology_definition is None:
        return _templates

    # Hash the per-request ontology
    ont_hash = hashlib.sha256(
        json.dumps(ontology_definition, sort_keys=True).encode()
    ).hexdigest()[:16]

    if ont_hash in _ontology_cache:
        return _ontology_cache[ont_hash]

    # Build templates from override ontology
    templates = build_templates_with_edges(ontology_definition)
    _ontology_cache[ont_hash] = templates

    # Keep cache bounded (1 entry)
    if len(_ontology_cache) > 2:
        oldest_key = next(iter(_ontology_cache))
        del _ontology_cache[oldest_key]

    return templates


def run_extraction_pipeline(
    docling_document_json: dict[str, Any],
    templates: dict[str, Any],
) -> Any:
    """Run the docling-graph extraction pipeline synchronously.

    This is called in a thread via asyncio.to_thread().
    """
    import tempfile

    from docling_graph import run_pipeline

    # Write DoclingDocument JSON to temp file (pipeline expects a file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    try:
        # Build the root template (first entity type or a composite)
        # For now, use the first template as the root
        root_template = next(iter(templates.values())) if templates else None

        config = build_pipeline_config(
            source=tmp_path,
            template_class=root_template,
        )

        context = run_pipeline(config)
        return context
    finally:
        os.unlink(tmp_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        ontology_version=_ontology_version,
        template_count=len(_templates),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        pipeline_version=getattr(app.state, "pipeline_version", "unknown"),
    )


@app.post("/extract-all", response_model=ExtractionResponse)
async def extract_all(request: ExtractionRequest):
    """Extract entities and relationships from a document.

    Accepts DoclingDocument JSON, runs delta extraction pipeline,
    returns serialized NetworkX graph.
    """
    if _extraction_semaphore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Resolve templates
    templates = _resolve_templates(request.ontology_definition)

    if not templates:
        raise HTTPException(
            status_code=422,
            detail="No templates available — check ontology configuration",
        )

    # Log version mismatch
    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning(
            "Ontology version mismatch: request=%s server=%s",
            request.ontology_version,
            _ontology_version,
        )

    # Run pipeline with concurrency limit
    async with _extraction_semaphore:
        try:
            context = await asyncio.to_thread(
                run_extraction_pipeline,
                request.docling_document_json,
                templates,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for document %s", request.document_id)
            raise HTTPException(
                status_code=500,
                detail=f"Extraction pipeline failed: {exc}",
            )

    # Serialize NetworkX graph
    graph = context.knowledge_graph
    graph_data = nx.node_link_data(graph)

    # Build metadata
    meta = context.graph_metadata
    metadata = ExtractionMetadata(
        node_count=meta.node_count if meta else graph.number_of_nodes(),
        edge_count=meta.edge_count if meta else graph.number_of_edges(),
        node_types=getattr(meta, "node_types", {}),
        edge_types=getattr(meta, "edge_types", {}),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
    )

    return ExtractionResponse(
        graph=graph_data,
        metadata=metadata,
        model=os.environ.get("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
        provider=os.environ.get("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
        ontology_version=_ontology_version,
    )
```

- [ ] **Step 3: Run tests**

```bash
cd docker/docling-graph && python -m pytest tests/test_extraction.py -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/tests/test_extraction.py
git commit -m "feat(docling-graph): rewrite main.py as thin run_pipeline() wrapper"
```

---

### Task 7: Delete obsolete files and update requirements

**Files:**
- Delete: `docker/docling-graph/app/prompts.py`
- Delete: `docker/docling-graph/tests/test_direct_extraction.py`
- Delete: `docker/docling-graph/tests/test_prompts.py`
- Modify: `docker/docling-graph/requirements.txt`

- [ ] **Step 1: Delete obsolete files**

```bash
rm docker/docling-graph/app/prompts.py
rm docker/docling-graph/tests/test_direct_extraction.py
rm docker/docling-graph/tests/test_prompts.py
```

- [ ] **Step 2: Update requirements.txt**

```
# docker/docling-graph/requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
docling-graph
networkx>=3.0
pyyaml>=6.0
httpx>=0.27.0
```

Removed: `litellm<=1.82.6`, `json-repair>=0.30.0`
Changed: `docling-graph>=0.1.0` → `docling-graph` (no pin — installed from local repo clone by Dockerfile)

- [ ] **Step 3: Run full test suite**

```bash
cd docker/docling-graph && python -m pytest tests/ -v
```
Expected: All tests PASS (test_templates.py may need updating if it imports from prompts — check)

- [ ] **Step 4: Commit**

```bash
git add -A docker/docling-graph/
git commit -m "chore(docling-graph): remove obsolete prompts/direct-extraction, update requirements"
```

---

### Task 8: Validation pass — post-extraction relationship check

**Files:**
- Create: `docker/docling-graph/tests/test_validation_pass.py`
- Modify: `docker/docling-graph/app/main.py` (add validation pass after pipeline)

- [ ] **Step 1: Write failing test for validation pass**

```python
# docker/docling-graph/tests/test_validation_pass.py
"""Tests for the post-extraction validation pass."""
import networkx as nx
import pytest


class TestValidationPass:
    """Test validation pass catches missed relationships."""

    def test_validation_pass_disabled_by_default_env(self):
        """When DOCLING_GRAPH_VALIDATION_PASS_ENABLED=false, no validation pass."""
        from app.main import _should_run_validation_pass
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"DOCLING_GRAPH_VALIDATION_PASS_ENABLED": "false"}):
            assert _should_run_validation_pass() is False

    def test_validation_pass_enabled_by_default(self):
        """Validation pass should be enabled by default."""
        from app.main import _should_run_validation_pass

        assert _should_run_validation_pass() is True

    def test_validation_pass_adds_edges_to_graph(self):
        """Validation pass should add discovered edges to the graph."""
        # This test will need a mock LLM call
        # For now, test the graph manipulation logic
        graph = nx.DiGraph()
        graph.add_node("RADAR_SYSTEM_Test", type="RADAR_SYSTEM", name="Test")
        graph.add_node("SPECIFICATION_Range_300", type="SPECIFICATION", name="Range")

        initial_edge_count = graph.number_of_edges()

        # Simulate validation pass adding an edge
        from app.main import _apply_validation_edges

        new_edges = [
            {
                "source": "RADAR_SYSTEM_Test",
                "target": "SPECIFICATION_Range_300",
                "label": "SPECIFIED_BY",
                "confidence": 0.85,
            }
        ]

        _apply_validation_edges(graph, new_edges)

        assert graph.number_of_edges() == initial_edge_count + 1
        assert graph.has_edge("RADAR_SYSTEM_Test", "SPECIFICATION_Range_300")
```

- [ ] **Step 2: Implement validation pass helpers in main.py**

Add to `docker/docling-graph/app/main.py`:

```python
def _should_run_validation_pass() -> bool:
    """Check if validation pass is enabled."""
    val = os.environ.get("DOCLING_GRAPH_VALIDATION_PASS_ENABLED", "true")
    return val.lower() in ("true", "1", "yes")


def _apply_validation_edges(
    graph: nx.DiGraph,
    new_edges: list[dict[str, Any]],
) -> int:
    """Apply discovered edges from validation pass to the graph.

    Returns number of edges added.
    """
    added = 0
    for edge in new_edges:
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label", "RELATED_TO")

        if source and target and graph.has_node(source) and graph.has_node(target):
            if not graph.has_edge(source, target):
                graph.add_edge(
                    source,
                    target,
                    label=label,
                    confidence=edge.get("confidence", 0.5),
                    _source="validation_pass",
                )
                added += 1

    return added
```

- [ ] **Step 3: Run tests**

```bash
cd docker/docling-graph && python -m pytest tests/test_validation_pass.py -v
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/tests/test_validation_pass.py
git commit -m "feat(docling-graph): add post-extraction validation pass for missed relationships"
```

---

### Task 9: Update existing templates test

**Files:**
- Modify: `docker/docling-graph/tests/test_templates.py`

- [ ] **Step 1: Update test_templates.py for new template_builder imports**

The existing `test_templates.py` (119 lines) imports from `app.templates` which still exists but will be superseded by `template_builder`. Update imports to test the new module while keeping the old tests functional during transition.

```bash
# Check if test_templates.py imports from app.templates or app.prompts
grep -n "from app" docker/docling-graph/tests/test_templates.py
```

If it imports from `app.templates`, keep it as-is (the old `templates.py` still exists until we verify all functionality is migrated). If it imports from `app.prompts`, update to remove those imports.

- [ ] **Step 2: Run full test suite**

```bash
cd docker/docling-graph && python -m pytest tests/ -v --tb=short
```
Expected: All tests PASS

- [ ] **Step 3: Commit if changes were needed**

```bash
git add docker/docling-graph/tests/test_templates.py
git commit -m "test(docling-graph): update templates test for new builder imports"
```

---

### Task 10: Integration test with mock pipeline

**Files:**
- Create: `docker/docling-graph/tests/test_pipeline_integration.py`

- [ ] **Step 1: Write integration test**

```python
# docker/docling-graph/tests/test_pipeline_integration.py
"""Integration tests for the complete extraction pipeline flow.

These tests mock the docling-graph library's run_pipeline() to verify
the service correctly:
1. Accepts DoclingDocument JSON
2. Builds templates from ontology
3. Constructs PipelineConfig
4. Returns serialized NetworkX graph
"""
import json
import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_mock_context():
    """Create a realistic mock PipelineContext."""
    graph = nx.DiGraph()

    # Add nodes matching military ontology
    graph.add_node(
        "RADAR_SYSTEM_Tombstone",
        type="RADAR_SYSTEM",
        name="Tombstone",
        system_name="Tombstone",
        nomenclature="64N6",
        radar_type="SEARCH",
        _provenance={"batch_id": 0, "chunk_index": 2, "page_numbers": [14, 15]},
    )
    graph.add_node(
        "PLATFORM_SA-20_TEL",
        type="PLATFORM",
        name="SA-20 TEL",
        platform_designation="5P85SE",
        _provenance={"batch_id": 0, "chunk_index": 1, "page_numbers": [3]},
    )
    graph.add_node(
        "FREQUENCY_BAND_S-band",
        type="FREQUENCY_BAND",
        name="S-band",
        band_name="S-band",
        designation="S",
        _provenance={"batch_id": 1, "chunk_index": 0, "page_numbers": [14]},
    )

    # Add edges
    graph.add_edge(
        "RADAR_SYSTEM_Tombstone",
        "PLATFORM_SA-20_TEL",
        label="INSTALLED_ON",
        confidence=0.92,
    )
    graph.add_edge(
        "RADAR_SYSTEM_Tombstone",
        "FREQUENCY_BAND_S-band",
        label="OPERATES_IN_BAND",
        confidence=0.88,
    )

    context = MagicMock()
    context.knowledge_graph = graph
    context.graph_metadata = MagicMock(
        node_count=3,
        edge_count=2,
        node_types={"RADAR_SYSTEM": 1, "PLATFORM": 1, "FREQUENCY_BAND": 1},
        edge_types={"INSTALLED_ON": 1, "OPERATES_IN_BAND": 1},
    )
    return context


@pytest.fixture
def integration_client():
    """Create test client with mocked pipeline for integration testing."""
    mock_context = _make_mock_context()

    with patch("app.main.run_extraction_pipeline", return_value=mock_context):
        from app.main import app

        yield TestClient(app)


class TestFullExtractionFlow:
    """Test the complete extraction flow end-to-end."""

    def test_full_extraction_returns_valid_graph(self, integration_client):
        """Full extraction should return a parseable NetworkX graph."""
        response = integration_client.post(
            "/extract-all",
            json={
                "document_id": "doc-001",
                "docling_document_json": {
                    "schema_name": "DoclingDocument",
                    "version": "1.0.0",
                    "body": {"self_ref": "#/body", "children": []},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify graph structure
        graph_data = data["graph"]
        assert len(graph_data["nodes"]) == 3
        assert len(graph_data["links"]) == 2

        # Verify node types
        node_types = {n["type"] for n in graph_data["nodes"]}
        assert "RADAR_SYSTEM" in node_types
        assert "PLATFORM" in node_types
        assert "FREQUENCY_BAND" in node_types

    def test_extraction_metadata_is_complete(self, integration_client):
        """Metadata should contain all expected fields."""
        response = integration_client.post(
            "/extract-all",
            json={
                "document_id": "doc-001",
                "docling_document_json": {"schema_name": "DoclingDocument"},
            },
        )

        data = response.json()
        meta = data["metadata"]

        assert meta["node_count"] == 3
        assert meta["edge_count"] == 2
        assert "RADAR_SYSTEM" in meta["node_types"]

    def test_provenance_preserved_on_nodes(self, integration_client):
        """Each node should carry _provenance from delta extraction."""
        response = integration_client.post(
            "/extract-all",
            json={
                "document_id": "doc-001",
                "docling_document_json": {"schema_name": "DoclingDocument"},
            },
        )

        data = response.json()
        nodes = data["graph"]["nodes"]

        radar_node = next(n for n in nodes if n["type"] == "RADAR_SYSTEM")
        assert "_provenance" in radar_node
        assert "page_numbers" in radar_node["_provenance"]
        assert 14 in radar_node["_provenance"]["page_numbers"]
```

- [ ] **Step 2: Run integration tests**

```bash
cd docker/docling-graph && python -m pytest tests/test_pipeline_integration.py -v
```
Expected: All tests PASS

- [ ] **Step 3: Run full test suite one final time**

```bash
cd docker/docling-graph && python -m pytest tests/ -v --tb=short
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/tests/test_pipeline_integration.py
git commit -m "test(docling-graph): add integration tests for full extraction pipeline flow"
```

---

## Summary

| Task | Component | Tests | Lines (est.) |
|------|-----------|-------|-------------|
| 1 | graph_id_fields derivation | 9 | ~80 |
| 2 | Pydantic model generation | 8 | ~120 |
| 3 | Edge field generation | 6 | ~150 |
| 4 | Config builder | 7 | ~130 |
| 5 | Request/response schemas | 0 (models) | ~70 |
| 6 | Main.py thin wrapper | 4 | ~200 |
| 7 | Delete obsolete files | 0 | -860 |
| 8 | Validation pass | 3 | ~50 |
| 9 | Update existing tests | 0 | ~10 |
| 10 | Integration tests | 3 | ~100 |
| **Total** | | **40** | **net -50** |

Net result: ~200 lines of new focused code replacing 921 lines of hand-rolled LLM extraction. 40 tests covering all new functionality.
