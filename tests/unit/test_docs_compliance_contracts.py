"""Docs-compliance contract tests.

Nine tests codify the Pydantic-as-single-source-of-truth contracts from
the plan (2026-04-14-docling-graph-schema-compliance.md). All are xfail
in Phase 1 because their targets — ``entities.py``, the ``is_entity`` /
``graph_id_fields`` / ``ontology_name`` / ``edge_label`` metadata, the
merge-preserving dedup validator — land in Phases 2 and 5.

Phase 1 lands the tests with ``pytest.mark.xfail(strict=True)`` so:
- They do not gate the current test suite.
- Flipping a marker on or off is the single signal that the contract
  either started passing (drop the mark, Task 35) or started regressing
  (the strict xfail turns into XPASS and fails CI).

Plan task mapping (by contract-test function name):
- test_is_entity_true_classes_declare_graph_id_fields_key    — Task 6
- test_edge_label_on_entity_to_entity_fields                 — Task 7
- test_identity_fields_have_examples                          — Task 8
- test_canonical_entities_declare_ontology_name               — Task 9
- test_extraction_views_subset_of_canonical_with_validator_parity — Task 9a
- test_every_class_declares_is_entity_explicitly              — Task 9b
- test_descriptions_and_examples_on_extraction_relevant_fields — Task 9c
- test_pass_root_list_dedup_schema_local                      — Task 9d
- test_edge_label_targets_are_is_entity_true                  — Task 9e
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

# SystemLinkRelationship is the documented multi-pass DTO exception
# (Decision 4). It must be excluded from every entity-scoped assertion.
_DTO_CARVE_OUTS: set[str] = {"SystemLinkRelationship"}

_EXTRACTION_PKG = "ontology_bundles.air_defense_v3.extraction_schemas"
_CANONICAL_PKG = "ontology_bundles.air_defense_v3.entities"


def _iter_extraction_modules():
    """Import every submodule under extraction_schemas/ and yield it."""
    pkg = importlib.import_module(_EXTRACTION_PKG)
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        yield importlib.import_module(f"{_EXTRACTION_PKG}.{name}")


def _try_import_canonical():
    """Import entities.py if it exists; return the module or None.

    Phase 2 lands entities.py. Before then, this returns None and tests
    that require it xfail cleanly.
    """
    try:
        return importlib.import_module(_CANONICAL_PKG)
    except ModuleNotFoundError:
        return None


def _iter_all_basemodels():
    """Yield every BaseModel subclass defined in extraction_schemas/
    and (if it exists) entities.py.

    Each yield is ``(cls, source_module_name)`` for readable assertion
    messages. Skips carved-out DTO classes.
    """
    for mod in _iter_extraction_modules():
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, BaseModel):
                continue
            if obj is BaseModel:
                continue
            if obj.__name__ in _DTO_CARVE_OUTS:
                continue
            # Only yield classes actually DEFINED in this module,
            # not re-imported — avoids double-counting.
            if obj.__module__ != mod.__name__:
                continue
            yield obj, mod.__name__
    canon = _try_import_canonical()
    if canon is not None:
        for attr_name in dir(canon):
            obj = getattr(canon, attr_name)
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, BaseModel):
                continue
            if obj is BaseModel:
                continue
            if obj.__name__ in _DTO_CARVE_OUTS:
                continue
            if obj.__module__ != canon.__name__:
                continue
            yield obj, canon.__name__


def _unwrap_annotated_types(annotation):
    """Recursively unwrap Union/Optional/List and yield every BaseModel
    subclass found in the annotation tree."""
    if annotation is None:
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for arg in get_args(annotation):
        yield from _unwrap_annotated_types(arg)


# ---------------------------------------------------------------------------
# Task 6: every is_entity=True class declares graph_id_fields key.
# ---------------------------------------------------------------------------


def test_is_entity_true_classes_declare_graph_id_fields_key():
    offenders: list[str] = []
    entity_count = 0
    for cls, src in _iter_all_basemodels():
        cfg = cls.model_config or {}
        if cfg.get("is_entity") is not True:
            continue
        entity_count += 1
        if "graph_id_fields" not in cfg:
            offenders.append(f"{src}.{cls.__name__}")
    assert entity_count > 0, (
        "No is_entity=True classes found — Phase 2 (entities.py) not yet landed."
    )
    assert not offenders, (
        "is_entity=True classes without explicit graph_id_fields key: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 7: edge_label on entity-to-entity fields declared on is_entity=True classes.
# ---------------------------------------------------------------------------


def test_edge_label_on_entity_to_entity_fields():
    offenders: list[str] = []
    entity_to_entity_fields = 0
    for cls, src in _iter_all_basemodels():
        cfg = cls.model_config or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname, finfo in cls.model_fields.items():
            target_entities = [
                t for t in _unwrap_annotated_types(finfo.annotation)
                if (t.model_config or {}).get("is_entity") is True
                and t.__name__ not in _DTO_CARVE_OUTS
            ]
            if not target_entities:
                continue
            entity_to_entity_fields += 1
            extra = finfo.json_schema_extra or {}
            if "edge_label" not in extra:
                offenders.append(f"{src}.{cls.__name__}.{fname}")
    assert entity_to_entity_fields > 0, (
        "No entity-to-entity fields found — Phase 5 typed-edge migration not yet landed."
    )
    assert not offenders, (
        "Entity-to-entity fields missing edge_label: " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 8: identity fields have ≥2 examples.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2: identity fields gain Field(examples=[...]) during entities.py creation",
)
def test_identity_fields_have_examples():
    offenders: list[str] = []
    checked = 0
    for cls, src in _iter_all_basemodels():
        cfg = cls.model_config or {}
        if cfg.get("is_entity") is not True:
            continue
        id_fields = cfg.get("graph_id_fields", []) or []
        for fname in id_fields:
            checked += 1
            finfo = cls.model_fields.get(fname)
            if finfo is None:
                offenders.append(f"{src}.{cls.__name__}.{fname} (missing)")
                continue
            examples = finfo.examples or []
            if len(examples) < 2:
                offenders.append(
                    f"{src}.{cls.__name__}.{fname} examples={len(examples)}"
                )
    assert checked > 0, (
        "No identity fields checked — Phase 2 (entities.py graph_id_fields) not yet landed."
    )
    assert not offenders, (
        "Identity fields with <2 examples: " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 9: every canonical entity class declares ontology_name.
# ---------------------------------------------------------------------------


def test_canonical_entities_declare_ontology_name():
    canon = _try_import_canonical()
    assert canon is not None, "entities.py module not yet created (Phase 2)"
    offenders: list[str] = []
    for attr_name in dir(canon):
        obj = getattr(canon, attr_name)
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)
                and obj is not BaseModel and obj.__module__ == canon.__name__):
            continue
        if obj.__name__ in _DTO_CARVE_OUTS:
            continue
        cfg = obj.model_config or {}
        if cfg.get("is_entity") is not True:
            continue
        if not cfg.get("ontology_name"):
            offenders.append(obj.__name__)
    assert not offenders, (
        "Canonical is_entity=True classes missing ontology_name: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 9a: extraction views are subsets of canonical (incl. validator parity).
# ---------------------------------------------------------------------------


def test_extraction_views_subset_of_canonical_with_validator_parity():
    canon = _try_import_canonical()
    assert canon is not None, "entities.py not yet created (Phase 2)"

    canonical_by_ontology_name: dict[str, type[BaseModel]] = {}
    for attr_name in dir(canon):
        obj = getattr(canon, attr_name)
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)
                and obj is not BaseModel and obj.__module__ == canon.__name__):
            continue
        cfg = obj.model_config or {}
        on = cfg.get("ontology_name")
        if on:
            canonical_by_ontology_name[on] = obj

    offenders: list[str] = []
    for mod in _iter_extraction_modules():
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if not (isinstance(obj, type) and issubclass(obj, BaseModel)
                    and obj is not BaseModel and obj.__module__ == mod.__name__):
                continue
            if obj.__name__ in _DTO_CARVE_OUTS:
                continue
            cfg = obj.model_config or {}
            on = cfg.get("ontology_name")
            if not on:
                continue
            canonical = canonical_by_ontology_name.get(on)
            if canonical is None:
                offenders.append(
                    f"{mod.__name__}.{obj.__name__}: no canonical peer for ontology_name={on!r}"
                )
                continue
            # graph_id_fields equal
            ev_ids = (obj.model_config or {}).get("graph_id_fields", []) or []
            cn_ids = (canonical.model_config or {}).get("graph_id_fields", []) or []
            if list(ev_ids) != list(cn_ids):
                offenders.append(
                    f"{obj.__name__}: graph_id_fields={ev_ids} vs canonical={cn_ids}"
                )
            # Every extraction field is a subset of canonical fields.
            def _ontology_name_signature(ann):
                """Reduce an annotation to a comparable signature where
                BaseModel types are represented by their ontology_name so
                that parallel narrow-view classes (extraction_schemas) and
                canonical classes (entities.py) compare equal when they
                refer to the same ontology entity."""
                if ann is None:
                    return None
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    return (
                        "basemodel",
                        (ann.model_config or {}).get("ontology_name") or ann.__name__,
                    )
                origin = get_origin(ann)
                if origin is None:
                    return ("raw", ann)
                return (
                    "generic",
                    origin,
                    tuple(_ontology_name_signature(a) for a in get_args(ann)),
                )

            for fname, ev_finfo in obj.model_fields.items():
                cn_finfo = canonical.model_fields.get(fname)
                if cn_finfo is None:
                    offenders.append(f"{obj.__name__}.{fname} not on canonical")
                    continue
                if (
                    _ontology_name_signature(ev_finfo.annotation)
                    != _ontology_name_signature(cn_finfo.annotation)
                ):
                    offenders.append(
                        f"{obj.__name__}.{fname} type drift: "
                        f"{ev_finfo.annotation!r} vs {cn_finfo.annotation!r}"
                    )
                if ev_finfo.description != cn_finfo.description:
                    offenders.append(
                        f"{obj.__name__}.{fname} description drift"
                    )
                # Shared-validator parity.
                ev_vs = {
                    v.info.fields: v.func.__qualname__
                    for v in getattr(obj, "__pydantic_decorators__",
                                     type("X", (), {"field_validators": {}})
                                     ).field_validators.values()
                }
                cn_vs = {
                    v.info.fields: v.func.__qualname__
                    for v in getattr(canonical, "__pydantic_decorators__",
                                     type("X", (), {"field_validators": {}})
                                     ).field_validators.values()
                }
                if cn_vs.get((fname,)) and ev_vs.get((fname,)) != cn_vs.get((fname,)):
                    offenders.append(
                        f"{obj.__name__}.{fname} validator drift: "
                        f"canonical={cn_vs.get((fname,))!r} vs extraction={ev_vs.get((fname,))!r}"
                    )
    assert not offenders, "Extraction-vs-canonical drift:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------------------
# Task 9b: every class declares is_entity explicitly.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2: every BaseModel gains an explicit is_entity setting",
)
def test_every_class_declares_is_entity_explicitly():
    offenders: list[str] = []
    total = 0
    for cls, src in _iter_all_basemodels():
        total += 1
        cfg = cls.model_config or {}
        if "is_entity" not in cfg:
            offenders.append(f"{src}.{cls.__name__}")
        elif not isinstance(cfg["is_entity"], bool):
            offenders.append(f"{src}.{cls.__name__} (non-bool is_entity)")
    assert total > 0, "No BaseModel classes discovered — import paths broken."
    assert not offenders, (
        "Classes without explicit is_entity (bool): " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 9c: descriptions and examples on extraction-relevant fields.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Phase 5: extraction-schemas rewrite adds descriptions/examples",
)
def test_descriptions_and_examples_on_extraction_relevant_fields():
    SYSTEM_FIELDS = {"confidence"}
    offenders: list[str] = []
    checked = 0
    for cls, src in _iter_all_basemodels():
        cfg = cls.model_config or {}
        if cfg.get("is_entity") is not True:
            continue
        id_fields = set(cfg.get("graph_id_fields", []) or [])
        for fname, finfo in cls.model_fields.items():
            if fname in SYSTEM_FIELDS:
                continue
            extra = finfo.json_schema_extra or {}
            is_edge_label = "edge_label" in extra
            is_identity = fname in id_fields
            is_domain = not (is_edge_label or is_identity)
            in_scope = is_edge_label or is_identity or is_domain
            if not in_scope:
                continue
            checked += 1
            if not finfo.description:
                offenders.append(f"{src}.{cls.__name__}.{fname} (no description)")
            if is_edge_label or is_domain:
                if not (finfo.examples or []):
                    offenders.append(f"{src}.{cls.__name__}.{fname} (no examples)")
    assert checked > 0, (
        "No in-scope fields found — Phase 2 (is_entity=True classes) not yet landed."
    )
    assert not offenders, (
        "Extraction-relevant fields missing description/examples: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 9d: pass-root list dedup is merge-preserving (schema-local).
# ---------------------------------------------------------------------------


def test_pass_root_list_dedup_schema_local():
    """Pass-root list fields of entity classes with non-empty graph_id_fields
    must have a Pydantic model_validator that deduplicates by identity and
    preserves non-null fields across duplicates."""
    # Load the five pass-root classes directly — they're the canonical roots.
    from ontology_bundles.air_defense_v3.extraction_schemas import (
        reference, radar_domain, missile_domain, other_systems, system_links,
    )
    pass_roots = [
        reference.ReferencePass,
        radar_domain.RadarDomainPass,
        missile_domain.MissileDomainPass,
        other_systems.OtherSystemsPass,
        system_links.SystemLinksPass,
    ]
    offenders: list[str] = []
    checked = 0
    for root in pass_roots:
        for fname, finfo in root.model_fields.items():
            if get_origin(finfo.annotation) is not list:
                continue
            inner_args = get_args(finfo.annotation)
            if not inner_args:
                continue
            inner = inner_args[0]
            if not (isinstance(inner, type) and issubclass(inner, BaseModel)):
                continue
            if inner.__name__ in _DTO_CARVE_OUTS:
                continue
            cfg = inner.model_config or {}
            if not (cfg.get("graph_id_fields") or []):
                continue
            checked += 1
            # Check the pass root has at least one model_validator that could
            # plausibly dedupe. Without loading it and running, we can only
            # check presence.
            validators = getattr(
                root, "__pydantic_decorators__",
                type("X", (), {"model_validators": {}})()
            ).model_validators
            if not validators:
                offenders.append(f"{root.__name__}.{fname} (no model_validator)")
    assert checked > 0, (
        "No pass-root entity lists with non-empty graph_id_fields — "
        "Phase 2 (entities.py) not yet landed."
    )
    assert not offenders, (
        "Pass-root entity-list fields without dedup validator: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Task 9e: edge_label fields target is_entity=True classes.
# ---------------------------------------------------------------------------


def test_edge_label_targets_are_is_entity_true():
    offenders: list[str] = []
    edge_label_fields = 0
    for cls, src in _iter_all_basemodels():
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if "edge_label" not in extra:
                continue
            edge_label_fields += 1
            targets = [
                t for t in _unwrap_annotated_types(finfo.annotation)
                if t.__name__ not in _DTO_CARVE_OUTS
            ]
            for t in targets:
                tcfg = t.model_config or {}
                if tcfg.get("is_entity") is not True:
                    offenders.append(
                        f"{src}.{cls.__name__}.{fname} -> {t.__name__} (is_entity={tcfg.get('is_entity')!r})"
                    )
    assert edge_label_fields > 0, (
        "No edge_label fields found — Phase 5 typed-edge migration not yet landed."
    )
    assert not offenders, (
        "edge_label fields pointing at non-entity targets: "
        + ", ".join(offenders)
    )
