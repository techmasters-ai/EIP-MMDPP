"""Manifest self-consistency checks.

These checks operate purely on the manifest dict structure (and coverage for
context), without loading any Pydantic classes.  They are collected into the
main error list by check_bundle().
"""
from __future__ import annotations

import importlib


def check_manifest_self_consistency(
    manifest: dict,
    coverage: dict,
    *,
    bundle_key: str,
) -> list[str]:
    errors: list[str] = []
    passes = manifest.get("passes", [])

    # 1. Unique pass names
    names = [p.get("name") for p in passes]
    dupes = [n for n in set(names) if names.count(n) > 1]
    for d in sorted(dupes):
        errors.append(f"{bundle_key}: duplicate pass name {d!r}")

    # 2. depends_on references only earlier passes (topological ordering)
    seen_names: set[str] = set()
    for p in passes:
        name = p.get("name")
        for dep in p.get("depends_on") or []:
            if dep not in seen_names:
                errors.append(
                    f"{bundle_key}: pass {name!r} depends_on {dep!r} "
                    f"which is not declared earlier in the pass list"
                )
        if name:
            seen_names.add(name)

    # 3. input_mode / entity-collection relationship
    #    - document_only: no required constraint on entity collections
    #    - document_plus_entity_refs: pass MUST have depends_on non-empty
    for p in passes:
        name = p.get("name")
        if p.get("input_mode") == "document_plus_entity_refs":
            if not p.get("depends_on"):
                errors.append(
                    f"{bundle_key}: pass {name!r} has input_mode="
                    f"document_plus_entity_refs but no depends_on"
                )

    # 4. entities are disjoint across passes EXCEPT for declared bridge entities
    entity_owners: dict[str, list[str]] = {}
    bridge_set: set[str] = set()
    for p in passes:
        name = p.get("name")
        for ent in p.get("primary_entity_types") or []:
            entity_owners.setdefault(ent, []).append(name)
        for ent in p.get("bridge_entity_types") or []:
            bridge_set.add(ent)
    for ent, owners in sorted(entity_owners.items()):
        if len(owners) > 1 and ent not in bridge_set:
            errors.append(
                f"{bundle_key}: entity {ent!r} is primary in multiple passes "
                f"({', '.join(owners)}) but not declared as a bridge"
            )

    # 5. Every template_class / module path is importable (smoke import)
    for p in passes:
        module_rel = p.get("module")
        cls_name = p.get("template_class")
        if not module_rel or not cls_name:
            errors.append(
                f"{bundle_key}: pass {p.get('name')!r} missing module or template_class"
            )
            continue
        full_module = f"ontology_bundles.{bundle_key}.{module_rel}"
        try:
            module = importlib.import_module(full_module)
            if not hasattr(module, cls_name):
                errors.append(
                    f"{bundle_key}: module {full_module} has no class {cls_name!r}"
                )
        except Exception as exc:
            errors.append(
                f"{bundle_key}: failed to import {full_module}: {exc}"
            )

    return errors
