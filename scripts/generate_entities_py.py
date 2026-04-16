"""Generate ontology_bundles/air_defense_v3/entities.py from ontology.yaml
+ the placement-table typed-edge map.

Plan v32 Phase 2 Tasks 14-19: this script produces the canonical Pydantic
entity classes. Run once, commit the output, then edit by hand if needed.
Parity tests (Phase 3 Task 20) catch drift; this is the ground-truth
source for Phase 1.

Usage:
    python3 scripts/generate_entities_py.py > ontology_bundles/air_defense_v3/entities.py
"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "ontology_bundles" / "air_defense_v3" / "ontology.yaml"

# Entity → layer map (stable import order; pass roots come last).
LAYERS = {
    1: [  # reference
        "DOCUMENT", "SECTION", "FIGURE", "TABLE", "SPREADSHEET",
    ],
    2: [  # military equipment
        "ORGANIZATION", "PLATFORM", "WEAPON_SYSTEM", "EQUIPMENT_SYSTEM",
        "SUBSYSTEM", "COMPONENT",
        "RADAR_SYSTEM", "MISSILE_SYSTEM", "AIR_DEFENSE_ARTILLERY_SYSTEM",
        "ELECTRONIC_WARFARE_SYSTEM", "FIRE_CONTROL_SYSTEM",
        "INTEGRATED_AIR_DEFENSE_SYSTEM", "LAUNCHER_SYSTEM",
    ],
    3: [  # EM/RF signal
        "FREQUENCY_BAND", "MODULATION", "RF_SIGNATURE", "RF_EMISSION", "WAVEFORM",
        "SCAN_PATTERN", "ANTENNA", "TRANSMITTER", "RECEIVER", "IF_AMPLIFIER",
        "SIGNAL_PROCESSING_CHAIN",
    ],
    4: [  # weapon/missile
        "GUIDANCE_METHOD", "SEEKER", "MISSILE_PERFORMANCE",
        "MISSILE_PHYSICAL_CHARACTERISTICS", "PROPULSION_STACK", "PROPULSION_STAGE",
    ],
    5: [  # operational
        "CAPABILITY", "RADAR_PERFORMANCE", "ENGAGEMENT_TIMELINE", "FORCE_STRUCTURE",
        "ASSEMBLY", "SPECIFICATION", "STANDARD", "PROCEDURE", "FAILURE_MODE",
        "TEST_EVENT",
    ],
}

# Known empty-identity anti-patterns (flagged in docstrings).
EMPTY_IDENTITY_ANTIPATTERN = {"PROPULSION_STACK"}


def entity_class_name(ontology_name: str) -> str:
    """RADAR_SYSTEM → RadarSystemEntity."""
    return "".join(p.capitalize() for p in ontology_name.lower().split("_")) + "Entity"


def snake(s: str) -> str:
    return s.lower()


def pluralize(s: str) -> str:
    if s.endswith("y"):
        return s[:-1] + "ies"
    if s.endswith("s"):
        return s + "_list"
    return s + "s"


# Placement rules mirroring docs/design/2026-04-14-relationship-placement-table.md
SINGLETON_RELS = {
    "HAS_GUIDANCE", "HAS_WARHEAD", "HAS_SEEKER", "HAS_FIRE_CONTROL",
    "HAS_COMMAND_POST", "HAS_LAUNCHER", "INSTALLED_ON", "MOUNTED_ON",
}

POST_MERGE_RELS = {
    "HAS_PROVENANCE", "MENTIONED_IN", "NEXT_CHUNK", "SAME_SECTION",
    "SAME_PAGE", "EXTRACTED_FROM",
}

SYSTEM_LINKS_RELS = {"ASSOCIATED_WITH", "CUES"}


def build_edge_fields_by_source(validation_matrix: list[dict]) -> dict[str, list[tuple]]:
    """Return {source_entity_name: [(rel, target, field_name, is_list), ...]}.
    Excludes post_merge and system_links rels."""
    out: dict[str, list[tuple]] = {}
    # Dedup: YAML has one duplicate triple
    seen = set()
    for t in validation_matrix:
        key = (t["source"], t["relationship"], t["target"])
        if key in seen:
            continue
        seen.add(key)
        r = t["relationship"]
        if r in POST_MERGE_RELS or r in SYSTEM_LINKS_RELS:
            continue
        src, tgt = t["source"], t["target"]
        is_list = r not in SINGLETON_RELS
        field = snake(tgt)
        if is_list:
            field = pluralize(field)
        out.setdefault(src, []).append((r, tgt, field, is_list))
    return out


def _python_type(prop_def: dict) -> str:
    t = prop_def.get("type", "string")
    if t == "integer":
        return "int"
    if t == "number":
        return "float"
    if t == "boolean":
        return "bool"
    if t == "array":
        return "list"
    return "str"


def format_field(name: str, prop_def: dict, is_identity: bool) -> str:
    py_type = _python_type(prop_def)
    description = (prop_def.get("description") or "").replace('"', '\\"')
    example = prop_def.get("example")
    enum = prop_def.get("enum")
    pattern = prop_def.get("pattern")

    kwargs = []
    if description:
        kwargs.append(f'description="{description}"')
    if example is not None:
        if is_identity:
            kwargs.append(f"examples=[{example!r}, {example!r}]")
        else:
            kwargs.append(f"examples=[{example!r}]")
    if pattern:
        kwargs.append(f"pattern={pattern!r}")
    if enum:
        enum_str = ", ".join(f'"{v}"' for v in enum)
        kwargs.append(f'json_schema_extra={{"enum": [{enum_str}]}}')

    if is_identity:
        ann = py_type
        args = "..., " + ", ".join(kwargs) if kwargs else "..."
    else:
        ann = f"Optional[{py_type}]"
        default = "None"
        args = (f"default={default}" + (", " + ", ".join(kwargs) if kwargs else ""))

    return f"    {name}: {ann} = Field({args})"


def build_class_source(et: dict, edge_fields: list[tuple], is_entity: bool) -> str:
    name = et["name"]
    class_name = entity_class_name(name)
    id_fields = et.get("identity_fields") or []
    id_scope = et.get("identity_scope", "global")
    parent = et.get("parent") or None
    description = (et.get("description") or "").replace('"""', "'''")
    props = (et.get("properties") or {}).get("properties") or {}

    # Docstring
    docstring_lines = [f'    """{et.get("label", name)} — {description}']
    if name in EMPTY_IDENTITY_ANTIPATTERN:
        docstring_lines.append("")
        docstring_lines.append(
            f"    KNOWN ANTI-PATTERN: ``{name}.graph_id_fields = []``. "
            "Every instance is a distinct node;"
        )
        docstring_lines.append(
            "    cannot dedup by logical identity. Flagged for Plan 2 cleanup."
        )
    docstring_lines.append('    """')
    docstring = "\n".join(docstring_lines)

    # model_config
    cfg_parts = [
        f'ontology_name="{name}"',
        f"graph_id_fields={id_fields!r}",
        f'identity_scope="{id_scope}"',
    ]
    if parent:
        cfg_parts.append(f'dodaf_parent="{parent}"')
    cfg_parts.append(f"is_entity={is_entity}")
    cfg_line = f"    model_config = ConfigDict({', '.join(cfg_parts)})"

    # Property fields (scalars from YAML)
    field_lines = []
    for fname, pdef in props.items():
        field_lines.append(format_field(fname, pdef, is_identity=(fname in id_fields)))

    # Edge fields (forward-ref strings for cross-layer types)
    for rel, target, field, is_list in edge_fields:
        target_class = entity_class_name(target)
        if is_list:
            ann = f'List["{target_class}"]'
            default = "default_factory=list"
        else:
            ann = f'Optional["{target_class}"]'
            default = "default=None"
        field_lines.append(
            f'    {field}: {ann} = edge(label="{rel}", {default})'
        )

    # System confidence — only emit if YAML does not already declare a
    # domain `confidence` property (e.g. ASSERTION). Tag with
    # ``json_schema_extra={"system_field": True}`` so introspection can
    # filter it out when producing a YAML-parity ``properties`` view.
    if is_entity and "confidence" not in props:
        field_lines.append(
            '    confidence: Optional[float] = Field('
            'default=None, '
            'description="Extraction confidence for this instance, 0–1.", '
            'ge=0.0, le=1.0, '
            'json_schema_extra={"system_field": True})'
        )

    if not field_lines:
        field_lines = ["    pass"]

    body = [f"class {class_name}(BaseModel):",
            docstring,
            cfg_line,
            "",
            *field_lines,
            ""]
    return "\n".join(body)


def main():
    with YAML_PATH.open() as f:
        ont = yaml.safe_load(f)
    by_name = {e["name"]: e for e in ont["entity_types"]}
    edges_by_src = build_edge_fields_by_source(ont["validation_matrix"])

    header = dedent('''\
        """Canonical Pydantic entity classes for the air_defense_v3 bundle.

        Plan v32 Phase 2 Tasks 14-19. Generated from ontology.yaml +
        docs/design/2026-04-14-relationship-placement-table.md via
        scripts/generate_entities_py.py.

        Every class declares:
        - ``model_config`` with ``ontology_name``, ``graph_id_fields``,
          ``identity_scope``, optional ``dodaf_parent``, and ``is_entity=True``.
        - Required identity fields (no default).
        - Optional non-identity fields with ``Field(description=..., examples=[...])``.
        - Typed-edge fields via ``edge(label=..., ...)`` for entity-to-entity
          relationships per the placement table.

        ``ALL_ENTITIES`` registry at the bottom exposes {ontology_name: class}
        for introspection consumers.

        Intra-pass relationship DTO classes (RadarRelationship, MissileRelationship,
        OtherSystemsRelationship) have been replaced by typed edges and are
        removed. ``SystemLinkRelationship`` remains as the multi-pass DTO
        exception per Decision 4 (lives in ``extraction_schemas/system_links.py``).
        """
        from __future__ import annotations

        from typing import Any, List, Optional

        from pydantic import BaseModel, ConfigDict, Field


        def edge(label: str, **field_kwargs: Any) -> Any:
            """Helper: declare a typed entity-to-entity edge field.

            Stores ``edge_label`` in ``json_schema_extra`` so introspection
            + contract tests can identify entity-to-entity fields. Otherwise
            delegates to ``Field(...)``.

            Args:
                label: The relationship type name (e.g. ``"HAS_ANTENNA"``).
                    Must match a member of ``RelationshipType``.
                **field_kwargs: Forwarded to ``Field`` (``default``,
                    ``default_factory``, ``description``, etc.).
            """
            existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
            existing_extra["edge_label"] = label
            return Field(json_schema_extra=existing_extra, **field_kwargs)
    ''')

    blocks = [header]

    # Emit in layer order for stable imports
    for layer_idx in sorted(LAYERS.keys()):
        blocks.append(f"\n# " + "-" * 70)
        blocks.append(f"# Layer {layer_idx}")
        blocks.append("# " + "-" * 70 + "\n")
        for name in LAYERS[layer_idx]:
            et = by_name.get(name)
            if et is None:
                continue
            edges = edges_by_src.get(name, [])
            blocks.append(build_class_source(et, edges, is_entity=True))

    # ALL_ENTITIES registry
    registry_entries = []
    for layer_idx in sorted(LAYERS.keys()):
        for name in LAYERS[layer_idx]:
            class_name = entity_class_name(name)
            registry_entries.append(f'    "{name}": {class_name},')
    registry = "\n".join([
        "",
        "# " + "-" * 70,
        "# ALL_ENTITIES registry",
        "# " + "-" * 70,
        "",
        "ALL_ENTITIES: dict[str, type[BaseModel]] = {",
        *registry_entries,
        "}",
        "",
        "# Rebuild forward references for all entity classes.",
        "for _cls in ALL_ENTITIES.values():",
        "    _cls.model_rebuild()",
        "",
    ])
    blocks.append(registry)

    sys.stdout.write("\n".join(blocks))


if __name__ == "__main__":
    main()
