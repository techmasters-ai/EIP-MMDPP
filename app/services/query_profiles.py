"""Registry-backed exact graph search using GraphStore traversal.

Replaces Cypher-template-based query generation with direct GraphStore
Protocol method calls for root resolution, section traversal, and evidence
attachment.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.query_profiles import QueryProfileRegistry
from app.schemas.graph_store import GraphEntityResult, GraphEvidenceItem
from app.schemas.query_profiles import (
    ActiveQueryProfilesResponse,
    QueryProfileDefinition,
    QueryProfileRegistryCreate,
    QueryProfileDossierResponse,
    QueryProfileDossierSection,
    QueryProfileRegistryResponse,
    QueryProfileSearchRequest,
    QueryProfileSectionResponse,
    QueryProfileStep,
    QueryProfileTraversal,
)
from app.services.ontology_templates import load_ontology, load_repository_ontology
from ontology_bundles.air_defense_v3.entities import (
    MissileSystemEntity,
    RadarSystemEntity,
)


# Service-side dispatch from entity_type string → canonical Pydantic
# class. Used by _project_field_groups to introspect which class's
# json_schema_extra to walk for a given resolved root. Kept in sync
# with app.schemas.query_profiles._CANONICAL_ROOT_ENTITY_TYPES via the
# contract test in tests/unit/test_query_profiles.py.
_CANONICAL_BY_ENTITY_TYPE: dict[str, type] = {
    "RADAR_SYSTEM": RadarSystemEntity,
    "MISSILE_SYSTEM": MissileSystemEntity,
}


def _canonical_class_for(entity_type: str):
    cls = _CANONICAL_BY_ENTITY_TYPE.get(entity_type)
    if cls is None:
        raise ValueError(
            f"No canonical Pydantic class registered for entity_type={entity_type!r}; "
            "section_properties profiles only run against types listed in "
            "_CANONICAL_BY_ENTITY_TYPE."
        )
    return cls


def _human_label(field_name: str) -> str:
    """Convert canonical field name to a display label (title-cased)."""
    return field_name.replace("_", " ").title()


def _project_field_groups(
    canonical_cls: type,
    instance_data: dict,
    profile_section: str,
):
    """Walk canonical_cls.model_fields, pick fields whose
    json_schema_extra['profile_sections'] contains *profile_section*,
    group by 'profile_subgroup'. Skip fields where instance_data[name]
    is None. Returns deterministically ordered groups (by subgroup
    name asc; fields by name asc within group). Spec §4.3.
    """
    from app.schemas.query_profiles import (
        QueryProfileFieldEntry, QueryProfileFieldGroup,
    )

    groups_by_subgroup: dict[str, list[QueryProfileFieldEntry]] = {}

    for fname, finfo in canonical_cls.model_fields.items():
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            continue
        sections = extra.get("profile_sections") or []
        if profile_section not in sections:
            continue
        value = instance_data.get(fname)
        if value is None:
            continue
        subgroup = extra.get("profile_subgroup") or ""
        entry = QueryProfileFieldEntry(
            name=fname,
            label=_human_label(fname),
            value=value,
            description=finfo.description,
            examples=list(finfo.examples) if finfo.examples else None,
            enum=extra.get("enum"),
        )
        groups_by_subgroup.setdefault(subgroup, []).append(entry)

    out: list[QueryProfileFieldGroup] = []
    for subgroup_key in sorted(groups_by_subgroup.keys()):
        entries = sorted(groups_by_subgroup[subgroup_key], key=lambda e: e.name)
        out.append(QueryProfileFieldGroup(
            subgroup=subgroup_key or None,
            subgroup_label=_human_label(subgroup_key) if subgroup_key else None,
            fields=entries,
        ))
    return out


_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_CURRENT_ONTOLOGY_NAME = "EIP-MMDPP Military Equipment & EM/RF Ontology"

_CURRENT_ROOT_ENTITY_TYPES = [
    "PLATFORM",
    "RADAR_SYSTEM",
    "MISSILE_SYSTEM",
    "AIR_DEFENSE_ARTILLERY_SYSTEM",
    "ELECTRONIC_WARFARE_SYSTEM",
    "FIRE_CONTROL_SYSTEM",
    "INTEGRATED_AIR_DEFENSE_SYSTEM",
    "LAUNCHER_SYSTEM",
    "WEAPON_SYSTEM",
    "EQUIPMENT_SYSTEM",
]

_CURRENT_RF_ENTITY_TYPES = [
    "FREQUENCY_BAND",
    "RF_EMISSION",
    "WAVEFORM",
    "MODULATION",
    "RF_SIGNATURE",
    "SCAN_PATTERN",
    "ANTENNA",
    "TRANSMITTER",
    "RECEIVER",
    "IF_AMPLIFIER",
    "SIGNAL_PROCESSING_CHAIN",
    "SEEKER",
    "SPECIFICATION",
]

_CURRENT_PERFORMANCE_ENTITY_TYPES = [
    "CAPABILITY",
    "RADAR_PERFORMANCE",
    "ENGAGEMENT_TIMELINE",
    "MISSILE_PERFORMANCE",
    "MISSILE_PHYSICAL_CHARACTERISTICS",
    "GUIDANCE_METHOD",
    "PROPULSION_STACK",
    "PROPULSION_STAGE",
    "SPECIFICATION",
    "STANDARD",
    "PROCEDURE",
    "FAILURE_MODE",
    "TEST_EVENT",
]

_CURRENT_RF_REL_TYPES = [
    "OPERATES_IN_BAND",
    "USES_WAVEFORM",
    "USES_MODULATION",
    "EMITS",
    "RADIATES",
    "RECEIVES",
    "HAS_SIGNATURE",
    "HAS_SCAN",
    "HAS_ANTENNA",
    "HAS_TRANSMITTER",
    "HAS_RECEIVER",
    "HAS_PROCESSING_CHAIN",
    "HAS_SEEKER",
    "SPECIFIED_BY",
]

_CURRENT_PERFORMANCE_REL_TYPES = [
    "HAS_PERFORMANCE",
    "PROVIDES",
    "SPECIFIED_BY",
    "TRACKS",
    "GUIDES",
    "DETECTS",
    "ENGAGES",
    "CUES",
    "DESIGNATES",
    "SUPPORTS_ENGAGEMENT_OF",
    "HAS_GUIDANCE",
    "HAS_TIMELINE",
]

_CURRENT_STRUCTURE_REL_TYPES = [
    "HAS_SUBSYSTEM",
    "HAS_COMPONENT",
    "HAS_STAGE",
]

_CURRENT_PART_REL_TYPES = [
    "PART_OF",
]


class QueryProfileRegistryNotFoundError(LookupError):
    """Raised when no active registry is available."""


class QueryProfileNotFoundError(LookupError):
    """Raised when a profile is not available in the active registry."""


class QueryRootNotFoundError(LookupError):
    """Raised when a requested root entity cannot be resolved."""


RegistryLike = QueryProfileRegistry | QueryProfileRegistryCreate


def _filter_known(items: list[str], allowed: set[str]) -> list[str]:
    return [item for item in items if item in allowed]


def _ontology_subset(*, repository_only: bool = False) -> dict[str, Any]:
    ontology = (
        load_repository_ontology()
        if repository_only
        else load_ontology()
    )
    return {
        "version": ontology.get("version"),
        "entity_types": ontology.get("entity_types", []),
        "relationship_types": ontology.get("relationship_types", []),
        "validation_matrix": ontology.get("validation_matrix", []),
    }


def build_default_registry_template() -> QueryProfileRegistryCreate:
    ontology = _ontology_subset(repository_only=True)
    entity_names = {
        item.get("name")
        for item in ontology.get("entity_types", [])
        if isinstance(item, dict) and item.get("name")
    }
    relationship_names = {
        item.get("name")
        for item in ontology.get("relationship_types", [])
        if isinstance(item, dict) and item.get("name")
    }

    root_types = _filter_known(_CURRENT_ROOT_ENTITY_TYPES, entity_names)
    rf_types = _filter_known(_CURRENT_RF_ENTITY_TYPES, entity_names)
    performance_types = _filter_known(_CURRENT_PERFORMANCE_ENTITY_TYPES, entity_names)
    structure_rels = _filter_known(_CURRENT_STRUCTURE_REL_TYPES, relationship_names)
    part_rels = _filter_known(_CURRENT_PART_REL_TYPES, relationship_names)
    rf_rels = _filter_known(_CURRENT_RF_REL_TYPES, relationship_names)
    performance_rels = _filter_known(_CURRENT_PERFORMANCE_REL_TYPES, relationship_names)

    profiles = [
        QueryProfileDefinition(
            id="system_dossier",
            label="System Dossier",
            description=(
                "Resolve a military system and return the configured exact graph sections "
                "for components, RF parameters, and performance characteristics."
            ),
            kind="dossier",
            exposed=True,
            root_entity_types=root_types,
            section_profile_ids=[
                "system_components",
                "system_rf_parameters",
                "system_performance",
            ],
            placeholder_query="e.g. AN/MPQ-65 or PAC-3 MSE",
        ),
        QueryProfileDefinition(
            id="system_components",
            label="System Components",
            description="Traverse subsystem/component structure and part hierarchy.",
            kind="section",
            exposed=True,
            root_entity_types=root_types,
            target_entity_types=[],
            traversals=[
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=structure_rels or _CURRENT_STRUCTURE_REL_TYPES,
                            min_hops=1,
                            max_hops=3,
                        )
                    ]
                ),
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="in",
                            rel_types=part_rels or _CURRENT_PART_REL_TYPES,
                            min_hops=1,
                            max_hops=3,
                        )
                    ]
                ),
            ],
            placeholder_query="e.g. AN/MPQ-65",
        ),
        QueryProfileDefinition(
            id="system_rf_parameters",
            label="System RF Parameters",
            description="Find band, waveform, emitter, receiver, seeker, and RF specification nodes.",
            kind="section",
            exposed=True,
            root_entity_types=root_types,
            target_entity_types=rf_types,
            traversals=[
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=rf_rels or _CURRENT_RF_REL_TYPES,
                            min_hops=1,
                            max_hops=2,
                        )
                    ]
                ),
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=structure_rels or _CURRENT_STRUCTURE_REL_TYPES,
                            min_hops=1,
                            max_hops=1,
                        ),
                        QueryProfileStep(
                            direction="out",
                            rel_types=rf_rels or _CURRENT_RF_REL_TYPES,
                            min_hops=1,
                            max_hops=2,
                        ),
                    ]
                ),
            ],
            placeholder_query="e.g. AN/MPQ-65",
        ),
        QueryProfileDefinition(
            id="system_performance",
            label="System Performance",
            description="Find performance, capability, engagement, guidance, and specification nodes.",
            kind="section",
            exposed=True,
            root_entity_types=root_types,
            target_entity_types=performance_types,
            traversals=[
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=performance_rels or _CURRENT_PERFORMANCE_REL_TYPES,
                            min_hops=1,
                            max_hops=2,
                        )
                    ]
                ),
                QueryProfileTraversal(
                    steps=[
                        QueryProfileStep(
                            direction="out",
                            rel_types=structure_rels or _CURRENT_STRUCTURE_REL_TYPES,
                            min_hops=1,
                            max_hops=1,
                        ),
                        QueryProfileStep(
                            direction="out",
                            rel_types=performance_rels or _CURRENT_PERFORMANCE_REL_TYPES,
                            min_hops=1,
                            max_hops=2,
                        ),
                    ]
                ),
            ],
            placeholder_query="e.g. PAC-3 MSE",
        ),
    ]

    return QueryProfileRegistryCreate(
        name="Current Military Systems Registry",
        description=(
            "Preloaded from the repository ontology and seeded with the deterministic "
            "system dossier/component/RF/performance exact graph query modes."
        ),
        ontology_name=_CURRENT_ONTOLOGY_NAME,
        ontology_version=str(ontology.get("version") or ""),
        ontology_definition=ontology,
        profiles=profiles,
        is_active=False,
    )


def registry_to_response(registry: QueryProfileRegistry) -> QueryProfileRegistryResponse:
    return QueryProfileRegistryResponse(
        id=registry.id,
        name=registry.name,
        description=registry.description,
        source_id=registry.source_id,
        ontology_name=registry.ontology_name,
        ontology_version=registry.ontology_version,
        ontology_definition=registry.ontology_definition,
        profiles=_profile_models(registry),
        is_active=registry.is_active,
        created_by=registry.created_by,
        created_at=registry.created_at,
        updated_at=registry.updated_at,
    )


def _profile_models(registry: RegistryLike) -> list[QueryProfileDefinition]:
    profiles = getattr(registry, "profiles", []) or []
    return [
        item
        if isinstance(item, QueryProfileDefinition)
        else QueryProfileDefinition.model_validate(item)
        for item in profiles
    ]


def active_registry_payload(
    registry: QueryProfileRegistry | None,
) -> ActiveQueryProfilesResponse:
    if registry is None:
        return ActiveQueryProfilesResponse(registry=None, exposed_profiles=[])

    payload = registry_to_response(registry)
    return ActiveQueryProfilesResponse(
        registry=payload,
        exposed_profiles=[profile for profile in payload.profiles if profile.exposed],
    )


async def get_active_registry(db: AsyncSession) -> QueryProfileRegistry | None:
    from sqlalchemy import select

    result = await db.execute(
        select(QueryProfileRegistry)
        .where(QueryProfileRegistry.is_active.is_(True))
        .order_by(QueryProfileRegistry.updated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_required_active_registry(db: AsyncSession) -> RegistryLike:
    registry = await get_active_registry(db)
    if registry is None:
        raise QueryProfileRegistryNotFoundError("No active query profile registry configured")
    return registry


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _select_best_candidate(
    candidates: list[Any],
    requested_name: str,
) -> Any | None:
    """Select the best candidate from a list of GraphEntityResult objects."""
    if not candidates:
        return None

    wanted = _normalize(requested_name)

    def _rank(candidate: Any) -> tuple[int, float]:
        name = _normalize(getattr(candidate, "name", "") or "")
        canonical = _normalize(getattr(candidate, "canonical_name", "") or "")
        exact = 1 if wanted in {name, canonical} else 0
        return exact, 0.0

    return max(candidates, key=_rank)


def _build_entity_result(entity: Any) -> GraphEntityResult:
    """Convert a GraphStore GraphEntityResult to a schema GraphEntityResult."""
    from app.services.graph_store import GraphEntityResult as StoreEntity
    if isinstance(entity, StoreEntity):
        return GraphEntityResult(
            node_id=entity.node_id,
            name=entity.name,
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            score=None,
            hop_count=None,
            relationship_types=[],
            properties=entity.properties,
        )
    # dict fallback
    return GraphEntityResult(
        node_id=entity.get("node_id", ""),
        name=entity.get("name", ""),
        entity_type=entity.get("entity_type", "UNKNOWN"),
        canonical_name=entity.get("canonical_name"),
        score=None,
        hop_count=None,
        relationship_types=[],
        properties=entity.get("properties", {}),
    )


def _merge_section_results(items: list[Any]) -> list[GraphEntityResult]:
    """Deduplicate and sort section results."""
    merged: dict[str, GraphEntityResult] = {}

    for item in items:
        entity = _build_entity_result(item)
        key = entity.node_id or f"{entity.entity_type}:{entity.name}"
        if key not in merged:
            merged[key] = entity

    return sorted(
        merged.values(),
        key=lambda item: (
            item.entity_type.casefold(),
            item.name.casefold(),
        ),
    )


def _profile_map(registry: RegistryLike) -> dict[str, QueryProfileDefinition]:
    return {profile.id: profile for profile in _profile_models(registry)}


def _get_profile(
    registry: RegistryLike,
    profile_id: str,
) -> QueryProfileDefinition:
    profile = _profile_map(registry).get(profile_id)
    if profile is None:
        raise QueryProfileNotFoundError(
            f"Profile '{profile_id}' is not defined in the active registry"
        )
    return profile


def _root_entity_types(
    profile: QueryProfileDefinition,
    profile_map: dict[str, QueryProfileDefinition],
) -> list[str]:
    if profile.root_entity_types:
        return profile.root_entity_types
    if profile.kind != "dossier":
        return []

    merged: list[str] = []
    seen: set[str] = set()
    for section_id in profile.section_profile_ids:
        section = profile_map.get(section_id)
        if section is None:
            continue
        for entity_type in section.root_entity_types:
            if entity_type not in seen:
                seen.add(entity_type)
                merged.append(entity_type)
    return merged


def _collect_rel_types(profile: QueryProfileDefinition) -> list[str]:
    """Collect all relationship types from a profile's traversals."""
    rel_types: list[str] = []
    seen: set[str] = set()
    for traversal in (profile.traversals or []):
        for step in traversal.steps:
            for rt in step.rel_types:
                if rt not in seen:
                    seen.add(rt)
                    rel_types.append(rt)
    return rel_types


def _max_depth(profile: QueryProfileDefinition) -> int:
    """Calculate max traversal depth from a profile's traversals."""
    max_d = 1
    for traversal in (profile.traversals or []):
        total = sum(step.max_hops for step in traversal.steps)
        max_d = max(max_d, total)
    return max_d


async def resolve_root_entity(
    graph_store: Any,
    registry: RegistryLike,
    profile: QueryProfileDefinition,
    request: QueryProfileSearchRequest,
) -> GraphEntityResult:
    """Resolve the root entity using GraphStore alias + fulltext search."""
    pmap = _profile_map(registry)
    root_types = _root_entity_types(profile, pmap)

    # 1. Alias search
    alias_matches = await graph_store.search_by_alias(
        request.query_text,
    )
    alias_filtered = [
        m for m in alias_matches
        if not root_types or getattr(m, "entity_type", "") in root_types
    ]

    # 2. Fulltext search
    fulltext_matches = await graph_store.fulltext_search(
        request.query_text,
        entity_types=root_types if root_types else None,
        limit=10,
    )

    all_matches = alias_filtered + fulltext_matches

    # 3. Co-extracted fallback: if best candidate has no relationships,
    # try co-extracted entities
    candidate = _select_best_candidate(all_matches, request.query_text)
    if candidate is None:
        # 3. Try direct name resolution
        resolved = await graph_store.resolve_root_entity(request.query_text)
        if resolved is not None:
            candidate = resolved
        else:
            # 4. Co-extracted fallback: find entities that co-occur with
            # a partial-match entity in the same source chunks
            try:
                co_extracted = await graph_store.get_co_extracted_entities(
                    request.query_text, limit=5,
                )
                co_filtered = [
                    e for e in co_extracted
                    if not root_types or getattr(e, "entity_type", "") in root_types
                ]
                candidate = _select_best_candidate(co_filtered, request.query_text)
            except Exception:
                candidate = None

            if candidate is None:
                raise QueryRootNotFoundError(
                    f"No matching root entity found for '{request.query_text}'"
                )

    resolved = _build_entity_result(candidate)

    if request.include_aliases and resolved.node_id:
        resolved.aliases = []

    return resolved


async def _fetch_section_items(
    graph_store: Any,
    resolved: GraphEntityResult,
    request: QueryProfileSearchRequest,
    profile: QueryProfileDefinition,
):
    """Fetch section items.

    Returns:
      - list[QueryProfileFieldGroup] when profile.kind == "section_properties"
      - list[GraphEntityResult]      when profile.kind == "section" (legacy)

    For section_properties profiles, resolves the root vertex's full property
    dict via get_entity_by_rid and projects it through _project_field_groups
    for each requested profile_section (spec §4.4).

    For legacy section profiles, compiles the traversal steps into a native
    ArcadeDB MATCH pattern and returns matching neighbor entities.
    """
    if profile.kind == "section_properties":
        if not resolved.node_id:
            return []
        instance_data = await graph_store.get_entity_by_rid(resolved.node_id)
        canonical = _canonical_class_for(resolved.entity_type)
        groups: list = []
        for section in profile.profile_sections:
            groups.extend(_project_field_groups(canonical, instance_data, section))
        return groups

    if not resolved.node_id:
        return []

    # QueryProfileDefinition.traversals is a list of traversals; previously
    # the code accessed `profile.traversal` (singular), which raised
    # AttributeError on Pydantic v2 because no such field exists. Run each
    # traversal as a separate directed MATCH and merge the results.
    traversals_with_steps = [
        t for t in (profile.traversals or []) if t.steps
    ]
    if not traversals_with_steps:
        # Fallback: generic undirected traversal if profile has no steps
        rel_types = _collect_rel_types(profile)
        depth = _max_depth(profile)
        neighbors = await graph_store.get_neighborhood(
            resolved.node_id,
            depth=depth,
            rel_types=rel_types if rel_types else None,
        )
        target_types = profile.target_entity_types or []
        if target_types:
            neighbors = [n for n in neighbors if getattr(n, "entity_type", "") in target_types]
        neighbors = [n for n in neighbors if getattr(n, "node_id", "") != resolved.node_id]
        return _merge_section_results(neighbors[:request.top_k])

    neighbors: list = []
    for traversal in traversals_with_steps:
        steps = [
            {
                "direction": step.direction,
                "rel_types": step.rel_types,
                "min_hops": step.min_hops,
                "max_hops": step.max_hops,
            }
            for step in traversal.steps
        ]
        rows = await graph_store.get_directed_traversal(
            resolved.node_id,
            steps=steps,
            target_entity_types=profile.target_entity_types or None,
            limit=request.top_k,
        )
        neighbors.extend(rows)

    # Exclude the root entity itself
    neighbors = [
        n for n in neighbors
        if getattr(n, "node_id", "") != resolved.node_id
    ]

    return _merge_section_results(neighbors[:request.top_k])


async def _fetch_chunk_evidence(
    db: AsyncSession,
    chunk_ids: list[str],
) -> dict[str, GraphEvidenceItem]:
    if not chunk_ids:
        return {}

    sql = text(
        """
        SELECT tc.id::text AS chunk_id,
               'text_chunk' AS chunk_type,
               tc.artifact_id,
               tc.document_id,
               d.filename AS document_name,
               tc.modality,
               tc.page_number,
               tc.classification,
               tc.chunk_text,
               d.document_metadata
        FROM retrieval.text_chunks tc
        JOIN ingest.documents d ON d.id = tc.document_id
        WHERE tc.id::text = ANY(:ids)
        UNION ALL
        SELECT ic.id::text AS chunk_id,
               'image_chunk' AS chunk_type,
               ic.artifact_id,
               ic.document_id,
               d.filename AS document_name,
               ic.modality,
               ic.page_number,
               ic.classification,
               ic.chunk_text,
               d.document_metadata
        FROM retrieval.image_chunks ic
        JOIN ingest.documents d ON d.id = ic.document_id
        WHERE ic.id::text = ANY(:ids)
        """
    )
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    evidence_map: dict[str, GraphEvidenceItem] = {}
    for row in rows:
        meta = row[9] if isinstance(row[9], dict) else {} if row[9] is None else __import__("json").loads(row[9])
        evidence_map[str(row[0])] = GraphEvidenceItem(
            chunk_id=row[0],
            chunk_type=row[1],
            artifact_id=row[2],
            document_id=row[3],
            document_name=row[4],
            modality=row[5],
            page_number=row[6],
            classification=row[7],
            content_text=row[8],
            source_characterization=meta.get("source_characterization"),
            date_of_information=meta.get("date_of_information"),
        )
    return evidence_map


async def attach_evidence(
    graph_store: Any,
    db: AsyncSession,
    items: list[GraphEntityResult],
    limit: int,
) -> None:
    """For each entity, look up EXTRACTED_FROM chunk refs and load evidence."""
    all_chunk_ids: list[str] = []
    entity_chunk_map: dict[str, list[str]] = {}

    for item in items:
        if not item.node_id:
            continue
        try:
            linked = await graph_store.get_entity_evidence_chunks(item.node_id, limit)
            chunk_ids = [
                r.get("chunk_id", "") if isinstance(r, dict) else getattr(r, "chunk_id", "")
                for r in linked[:limit]
            ]
            chunk_ids = [c for c in chunk_ids if c]
            entity_chunk_map[item.node_id] = chunk_ids
            all_chunk_ids.extend(chunk_ids)
        except Exception:
            entity_chunk_map[item.node_id] = []

    chunk_map = await _fetch_chunk_evidence(db, list(dict.fromkeys(all_chunk_ids)))

    for item in items:
        if not item.node_id:
            continue
        chunk_ids = entity_chunk_map.get(item.node_id, [])
        item.evidence = [
            chunk_map[cid]
            for cid in chunk_ids
            if cid in chunk_map
        ]


async def execute_section_search(
    graph_store: Any,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    registry: RegistryLike | None = None,
) -> QueryProfileSectionResponse:
    registry = registry or await get_required_active_registry(db)
    profile = _get_profile(registry, request.profile_id)
    if profile.kind not in ("section", "section_properties"):
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a section query profile"
        )

    resolved = await resolve_root_entity(graph_store, registry, profile, request)
    raw = await _fetch_section_items(graph_store, resolved, request, profile)

    if profile.kind == "section_properties":
        field_groups = raw  # list[QueryProfileFieldGroup]
        related_systems: list[GraphEntityResult] = []
        if profile.include_associated_systems and resolved.node_id:
            related_systems = await graph_store.get_associated_systems(resolved.node_id)
        if request.include_evidence and related_systems:
            await attach_evidence(graph_store, db, [resolved] + related_systems, request.evidence_top_k)
        total = sum(len(g.fields) for g in field_groups) + len(related_systems)
        return QueryProfileSectionResponse(
            registry_id=getattr(registry, "id", None),
            profile_id=profile.id,
            profile_label=profile.label,
            resolved_root=resolved,
            field_groups=field_groups,
            related_systems=related_systems,
            items=[],
            total=total,
        )

    items = raw  # list[GraphEntityResult]
    if request.include_evidence:
        await attach_evidence(graph_store, db, [resolved] + items, request.evidence_top_k)

    return QueryProfileSectionResponse(
        registry_id=getattr(registry, "id", None),
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        items=items,
        total=len(items),
    )


async def execute_dossier_search(
    graph_store: Any,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    registry: RegistryLike | None = None,
) -> QueryProfileDossierResponse:
    registry = registry or await get_required_active_registry(db)
    pmap = _profile_map(registry)
    profile = _get_profile(registry, request.profile_id)
    if profile.kind != "dossier":
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a dossier query profile"
        )

    resolved = await resolve_root_entity(graph_store, registry, profile, request)

    sections: list[QueryProfileDossierSection] = []
    all_items: list[GraphEntityResult] = [resolved]
    for section_id in profile.section_profile_ids:
        section_profile = pmap.get(section_id)
        if section_profile is None:
            continue
        if section_profile.kind not in ("section", "section_properties"):
            continue
        raw = await _fetch_section_items(graph_store, resolved, request, section_profile)

        if section_profile.kind == "section_properties":
            field_groups = raw  # list[QueryProfileFieldGroup]
            related_systems: list[GraphEntityResult] = []
            if section_profile.include_associated_systems and resolved.node_id:
                related_systems = await graph_store.get_associated_systems(resolved.node_id)
            all_items.extend(related_systems)
            total = sum(len(g.fields) for g in field_groups) + len(related_systems)
            sections.append(
                QueryProfileDossierSection(
                    profile_id=section_profile.id,
                    profile_label=section_profile.label,
                    kind="section_properties",
                    field_groups=field_groups,
                    related_systems=related_systems,
                    items=[],
                    total=total,
                )
            )
        else:
            items = raw  # list[GraphEntityResult]
            all_items.extend(items)
            sections.append(
                QueryProfileDossierSection(
                    profile_id=section_profile.id,
                    profile_label=section_profile.label,
                    kind="section",
                    items=items,
                    total=len(items),
                )
            )

    if request.include_evidence:
        await attach_evidence(graph_store, db, all_items, request.evidence_top_k)

    return QueryProfileDossierResponse(
        registry_id=getattr(registry, "id", None),
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        aliases=resolved.aliases if request.include_aliases else [],
        sections=sections,
    )
