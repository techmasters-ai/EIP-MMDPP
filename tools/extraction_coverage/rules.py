"""Extraction coverage checker — 13 active rules + check_bundle composer.

Rule numbering matches spec §2.  Rules 7 and 14 are intentionally absent:
  - Rule 7: removed from spec prior to implementation.
  - Rule 14: rejection-reason coverage is enforced by a pytest test landing
    in Chunk 3 (test_extraction_merge.py); do NOT implement here.
"""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import get_args, get_origin

import yaml
from pydantic import BaseModel

from ontology_bundles._shared.limits import DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS
from tools.extraction_coverage.manifest_consistency import check_manifest_self_consistency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_FIELDS: frozenset[str] = frozenset({"confidence"})

SCHEMA_SIZE_THRESHOLD_CHARS: int = DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS

# Override table for class names that don't map cleanly via the generic rule.
_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "SPCEntity": "SIGNAL_PROCESSING_CHAIN",
    "ADAEntity": "AIR_DEFENSE_ARTILLERY_SYSTEM",
    "EWSystemEntity": "ELECTRONIC_WARFARE_SYSTEM",
    "FireControlSystemEntity": "FIRE_CONTROL_SYSTEM",
    "WeaponSystemEntity": "WEAPON_SYSTEM",
    "IADSEntity": "INTEGRATED_AIR_DEFENSE_SYSTEM",
}


# ---------------------------------------------------------------------------
# Helpers shared by rules 6 and 8 (copied from tests/ — do NOT import there)
# ---------------------------------------------------------------------------

def _unwrap_models(annotation) -> list[type[BaseModel]]:
    """Return all BaseModel subclasses reachable via annotation type args."""
    if annotation is None:
        return []
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return []
    results: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        results.extend(_unwrap_models(arg))
    return results


def _iter_nested_models(model_cls: type[BaseModel]):
    """Walk model fields recursively, yielding every nested BaseModel."""
    seen = {model_cls}
    stack = [model_cls]
    while stack:
        cls = stack.pop()
        for field_info in cls.model_fields.values():
            for nested in _unwrap_models(field_info.annotation):
                if nested not in seen:
                    seen.add(nested)
                    stack.append(nested)
                    yield nested


# ---------------------------------------------------------------------------
# Rule 8 helpers
# ---------------------------------------------------------------------------

def _class_name_to_ontology_name(class_name: str) -> str | None:
    """Convert a Python class name to its ontology entity name, or None for
    relationship/link classes."""
    if "Relationship" in class_name or "Link" in class_name:
        return None
    if class_name in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[class_name]
    # Generic rule: strip trailing Entity, then CamelCase → UPPER_SNAKE_CASE
    base = class_name.removesuffix("Entity")
    # Insert underscores before uppercase letters following a lowercase letter
    snake = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", base)
    # Insert underscores before uppercase sequences followed by a lowercase letter
    snake = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", snake)
    return snake.upper()


def _extract_ontology_properties(ontology: dict, entity_name: str) -> set[str]:
    """Return the set of property names for a named ontology entity."""
    for entity_def in ontology.get("entity_types", []):
        if not isinstance(entity_def, dict) or entity_def.get("name") != entity_name:
            continue
        props_field = entity_def.get("properties")
        if isinstance(props_field, dict):
            # Nested dict form: {"type": "object", "properties": {"field": ...}}
            inner = props_field.get("properties")
            if isinstance(inner, dict):
                return set(inner.keys())
            # Direct dict of prop names (fallback)
            return set(props_field.keys())
        if isinstance(props_field, list):
            return {p for p in props_field if isinstance(p, str)}
    return set()


# ---------------------------------------------------------------------------
# Rule 1 — coverage.extract ∪ derive ⊆ ontology.entity_types
# ---------------------------------------------------------------------------

def _check_coverage_subset(ontology: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    ontology_types = {
        e["name"] for e in ontology.get("entity_types", [])
        if isinstance(e, dict) and "name" in e
    }
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    derive_set = set(coverage.get("entity_types", {}).get("derive", []) or [])
    for missing in sorted((extract_set | derive_set) - ontology_types):
        errors.append(
            f"coverage declares entity {missing!r} that is not in ontology.entity_types"
        )
    return errors


# ---------------------------------------------------------------------------
# Rule 2 — manifest entity_types in coverage.extract
# ---------------------------------------------------------------------------

def _check_manifest_entities_in_coverage(manifest: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    for pass_def in manifest.get("passes", []):
        declared = (
            set(pass_def.get("primary_entity_types") or [])
            | set(pass_def.get("bridge_entity_types") or [])
        )
        for missing in sorted(declared - extract_set):
            errors.append(
                f"pass {pass_def.get('name')!r} declares entity {missing!r} "
                f"not in coverage.entity_types.extract"
            )
    return errors


# ---------------------------------------------------------------------------
# Rule 3 — manifest relationship_types in coverage.extract
# ---------------------------------------------------------------------------

def _check_manifest_relationships_in_coverage(manifest: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    extract_set = set(coverage.get("relationship_types", {}).get("extract", []) or [])
    for pass_def in manifest.get("passes", []):
        declared = set(pass_def.get("extracted_relationship_types") or [])
        for missing in sorted(declared - extract_set):
            errors.append(
                f"pass {pass_def.get('name')!r} declares relationship {missing!r} "
                f"not in coverage.relationship_types.extract"
            )
    return errors


# ---------------------------------------------------------------------------
# Rule 4 — extracted relationships have validation_matrix rows
# ---------------------------------------------------------------------------

def _check_relationships_in_validation_matrix(manifest: dict, ontology: dict) -> list[str]:
    errors: list[str] = []
    matrix_rels = {
        row.get("relationship")
        for row in ontology.get("validation_matrix", [])
        if isinstance(row, dict) and row.get("relationship")
    }
    declared: set[str] = set()
    for pass_def in manifest.get("passes", []):
        declared |= set(pass_def.get("extracted_relationship_types") or [])
    for rel in sorted(declared - matrix_rels):
        errors.append(
            f"relationship {rel!r} is extracted but has no row in ontology.validation_matrix"
        )
    return errors


# ---------------------------------------------------------------------------
# Rule 5 — pass schema size under threshold
# ---------------------------------------------------------------------------

def _check_schema_size(template_cls: type[BaseModel]) -> list[str]:
    schema_str = json.dumps(template_cls.model_json_schema())
    if len(schema_str) > SCHEMA_SIZE_THRESHOLD_CHARS:
        return [
            f"{template_cls.__name__}: JSON schema size "
            f"{len(schema_str)} > threshold {SCHEMA_SIZE_THRESHOLD_CHARS}"
        ]
    return []


# ---------------------------------------------------------------------------
# Rule 6 — recursive partial-safety (every non-system field must be Optional)
# ---------------------------------------------------------------------------

def _check_recursive_partial_safety(template_cls: type[BaseModel]) -> list[str]:
    errors: list[str] = []
    for model in [template_cls, *_iter_nested_models(template_cls)]:
        for field_name, field_info in model.model_fields.items():
            if field_name in SYSTEM_FIELDS:
                continue
            # In Pydantic v2, field_info.is_required() is the authoritative
            # check.  field_info.default is PydanticUndefined (not None) for
            # required fields, so `default is not None` cannot be used to detect
            # optional-ness — it would always be True for required fields too.
            if field_info.is_required():
                errors.append(
                    f"{model.__name__}.{field_name} is required; "
                    f"extraction models must tolerate partial LLM output"
                )
    return errors


# Rule 7: intentionally removed from spec prior to implementation.


# ---------------------------------------------------------------------------
# Rule 8 — extraction field names ⊆ ontology entity properties
# ---------------------------------------------------------------------------

def _check_extraction_subset_of_ontology(
    template_cls: type[BaseModel], ontology: dict,
) -> list[str]:
    errors: list[str] = []
    for model in _iter_nested_models(template_cls):
        ontology_name = _class_name_to_ontology_name(model.__name__)
        if ontology_name is None:
            continue  # Relationship / Link class — skip
        ontology_props = _extract_ontology_properties(ontology, ontology_name)
        if not ontology_props:
            # Unknown entity name — rule 2 will catch the actual violation
            continue
        for field_name in model.model_fields:
            if field_name in SYSTEM_FIELDS:
                continue
            if field_name not in ontology_props:
                errors.append(
                    f"{model.__name__}.{field_name} is not a property of "
                    f"ontology entity {ontology_name}"
                )
    return errors


# ---------------------------------------------------------------------------
# Rule 9 — identity_fields declared on every extract-bucket entity
# ---------------------------------------------------------------------------

def _check_identity_fields_completeness(ontology: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    by_name = {
        e.get("name"): e for e in ontology.get("entity_types", [])
        if isinstance(e, dict) and "name" in e
    }
    for name in sorted(extract_set):
        entity = by_name.get(name)
        if entity is None:
            continue  # Rule 1 catches this
        if "identity_fields" not in entity:
            errors.append(f"{name}: missing identity_fields declaration")
    return errors


# ---------------------------------------------------------------------------
# Rule 10 — display label computable (identity_fields or properties non-empty)
# ---------------------------------------------------------------------------

def _check_display_label_present(ontology: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    by_name = {
        e.get("name"): e for e in ontology.get("entity_types", [])
        if isinstance(e, dict) and "name" in e
    }
    for name in sorted(extract_set):
        entity = by_name.get(name)
        if entity is None:
            continue
        identity_fields = entity.get("identity_fields") or []
        props = _extract_ontology_properties(ontology, name)
        if not identity_fields and not props:
            errors.append(
                f"{name}: has empty identity_fields AND no properties — "
                f"build_display_label cannot produce a label"
            )
    return errors


# ---------------------------------------------------------------------------
# Rule 11 — identity_scope ∈ {"document", "global"}
# ---------------------------------------------------------------------------

def _check_identity_scope_required(ontology: dict, coverage: dict) -> list[str]:
    errors: list[str] = []
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    by_name = {
        e.get("name"): e for e in ontology.get("entity_types", [])
        if isinstance(e, dict) and "name" in e
    }
    for name in sorted(extract_set):
        entity = by_name.get(name)
        if entity is None:
            continue
        scope = entity.get("identity_scope")
        if scope not in ("document", "global"):
            errors.append(
                f"{name}: identity_scope must be 'document' or 'global' (got {scope!r})"
            )
    return errors


# ---------------------------------------------------------------------------
# Rule 12 — [WARN] identity_fields=[] + identity_scope=global
# ---------------------------------------------------------------------------

def _check_empty_identity_global_warning(ontology: dict, coverage: dict) -> list[str]:
    warnings: list[str] = []
    extract_set = set(coverage.get("entity_types", {}).get("extract", []) or [])
    by_name = {
        e.get("name"): e for e in ontology.get("entity_types", [])
        if isinstance(e, dict) and "name" in e
    }
    for name in sorted(extract_set):
        entity = by_name.get(name)
        if entity is None:
            continue
        if entity.get("identity_fields") == [] and entity.get("identity_scope") == "global":
            warnings.append(
                f"{name}: identity_fields=[] combined with identity_scope=global "
                f"makes every instance collapse to one global vertex via content-hash"
            )
    return warnings


# ---------------------------------------------------------------------------
# Rule 13 — bridge entities have exactly one ontology declaration
# ---------------------------------------------------------------------------

def _check_bridge_scope_consistency(
    ontology: dict, coverage: dict, manifest: dict,
) -> list[str]:
    errors: list[str] = []
    bridge_entities: set[str] = set()
    for pass_def in manifest.get("passes", []):
        bridge_entities |= set(pass_def.get("bridge_entity_types") or [])

    by_name: dict[str, list[dict]] = {}
    for e in ontology.get("entity_types", []):
        if isinstance(e, dict) and "name" in e:
            by_name.setdefault(e["name"], []).append(e)

    for bridge_name in sorted(bridge_entities):
        entries = by_name.get(bridge_name, [])
        if len(entries) == 0:
            errors.append(
                f"bridge entity {bridge_name!r} not declared in ontology.entity_types"
            )
        elif len(entries) > 1:
            errors.append(
                f"bridge entity {bridge_name!r} has {len(entries)} declarations "
                f"in ontology.entity_types (must be exactly one)"
            )
    return errors


# Rule 14: intentionally not implemented here.  Rejection-reason coverage is
# enforced by a pytest test in Chunk 3 (test_extraction_merge.py).


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

def _load_pass_template_by_manifest(
    bundle_key: str, pass_def: dict,
) -> type[BaseModel]:
    """Resolve manifest 'module' + 'template_class' into a Pydantic class.

    manifest.yaml stores 'module' as a bundle-relative dotted path like
    'extraction_schemas.radar_domain'.  The absolute import path is
    'ontology_bundles.<bundle_key>.<module>'.
    """
    module_rel = pass_def["module"]
    class_name = pass_def["template_class"]
    full_module = f"ontology_bundles.{bundle_key}.{module_rel}"
    module = importlib.import_module(full_module)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# check_bundle — composer
# ---------------------------------------------------------------------------

def check_bundle(bundle_path: Path) -> tuple[list[str], list[str]]:
    """Run all rules against the bundle at *bundle_path*.

    Returns (errors, warnings).  An empty errors list means the bundle passes.
    """
    errors: list[str] = []
    warnings: list[str] = []

    ontology = yaml.safe_load((bundle_path / "ontology.yaml").read_text())
    manifest = yaml.safe_load((bundle_path / "manifest.yaml").read_text())
    coverage = yaml.safe_load((bundle_path / "coverage.yaml").read_text())
    bundle_key = bundle_path.name

    errors += _check_coverage_subset(ontology, coverage)
    errors += _check_manifest_entities_in_coverage(manifest, coverage)
    errors += _check_manifest_relationships_in_coverage(manifest, coverage)
    errors += _check_relationships_in_validation_matrix(manifest, ontology)
    errors += _check_identity_fields_completeness(ontology, coverage)
    errors += _check_display_label_present(ontology, coverage)
    errors += _check_identity_scope_required(ontology, coverage)
    errors += _check_bridge_scope_consistency(ontology, coverage, manifest)
    warnings += _check_empty_identity_global_warning(ontology, coverage)

    for pass_def in manifest.get("passes", []):
        try:
            template_cls = _load_pass_template_by_manifest(bundle_key, pass_def)
        except (ImportError, AttributeError) as exc:
            errors.append(
                f"pass {pass_def.get('name', '?')}: "
                f"cannot import {pass_def.get('module')}."
                f"{pass_def.get('template_class')}: {exc}"
            )
            continue
        errors += _check_schema_size(template_cls)
        errors += _check_recursive_partial_safety(template_cls)
        errors += _check_extraction_subset_of_ontology(template_cls, ontology)

    errors += check_manifest_self_consistency(manifest, coverage, bundle_key=bundle_key)

    return errors, warnings
