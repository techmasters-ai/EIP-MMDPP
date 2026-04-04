"""Registry-backed exact graph search using fixed Cypher template generation."""

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

_ALIAS_QUERY = """
    MATCH (node:Entity)-[:HAS_ALIAS]->(a:Alias)
    WHERE toLower(a.alias_name) = toLower($query)
      AND (size($root_types) = 0 OR node.entity_type IN $root_types)
    OPTIONAL MATCH (node)-[r]->(:Entity)
    WITH node, count(r) AS rel_count
    RETURN node.id AS node_id,
           node.name AS name,
           node.canonical_name AS canonical_name,
           node.entity_type AS entity_type,
           properties(node) AS properties,
           100.0 AS score,
           rel_count
    LIMIT 10
"""

_FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
    YIELD node, score
    WHERE size($root_types) = 0 OR node.entity_type IN $root_types
    OPTIONAL MATCH (node)-[r]->(:Entity)
    WITH node, score, count(r) AS rel_count
    RETURN node.id AS node_id,
           node.name AS name,
           node.canonical_name AS canonical_name,
           node.entity_type AS entity_type,
           properties(node) AS properties,
           score,
           rel_count
    LIMIT 10
"""

_ALIAS_OF_QUERY = """
    MATCH (other:Entity)
    WHERE toLower(other.name) = toLower($query)
    MATCH (node:Entity)-[:ALIAS_OF]-(other)
    WHERE size($root_types) = 0 OR node.entity_type IN $root_types
    OPTIONAL MATCH (node)-[r]->(:Entity)
    WITH node, count(r) AS rel_count
    RETURN node.id AS node_id,
           node.name AS name,
           node.canonical_name AS canonical_name,
           node.entity_type AS entity_type,
           properties(node) AS properties,
           99.0 AS score,
           rel_count
    LIMIT 10
"""

_ALIASES_BY_NODE_ID_QUERY = """
    MATCH (node:Entity {id: $node_id})
    OPTIONAL MATCH (node)-[:HAS_ALIAS]->(a:Alias)
    RETURN collect(DISTINCT a.alias_name) AS aliases
"""

_COEXTRACTED_FALLBACK_QUERY = """
    MATCH (origin:Entity {id: $node_id})-[:EXTRACTED_FROM]->(c:ChunkRef)<-[:EXTRACTED_FROM]-(sibling:Entity)
    WHERE sibling.id <> $node_id
      AND (size($root_types) = 0 OR sibling.entity_type IN $root_types)
    WITH sibling, count(DISTINCT c) AS shared_chunks
    OPTIONAL MATCH (sibling)-[r]->(m:Entity)
    WITH sibling, shared_chunks, count(r) AS rel_count
    WHERE rel_count > 0
    RETURN sibling.id AS node_id,
           sibling.name AS name,
           sibling.canonical_name AS canonical_name,
           sibling.entity_type AS entity_type,
           properties(sibling) AS properties,
           toFloat(shared_chunks) AS score,
           rel_count
    ORDER BY rel_count DESC, shared_chunks DESC
    LIMIT 5
"""

_EVIDENCE_REFS_QUERY = """
    UNWIND $entity_ids AS entity_id
    MATCH (n:Entity {id: entity_id})
    OPTIONAL MATCH (n)-[:EXTRACTED_FROM]->(c:ChunkRef)
    WITH entity_id,
         [ref IN collect({chunk_id: c.chunk_id, chunk_type: c.chunk_type}) WHERE ref.chunk_id IS NOT NULL][..$limit] AS refs
    RETURN entity_id, refs
"""


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


async def _run_neo4j_query(driver, cypher: str, **params) -> list[dict[str, Any]]:
    async with driver.session() as session:
        result = await session.run(cypher, parameters=params)
        return await result.data()


def _select_best_candidate(
    candidates: list[dict[str, Any]],
    requested_name: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    wanted = _normalize(requested_name)

    def _rank(candidate: dict[str, Any]) -> tuple[int, int, float]:
        name = _normalize(str(candidate.get("name", "")))
        canonical = _normalize(str(candidate.get("canonical_name", "") or ""))
        exact = 1 if wanted in {name, canonical} else 0
        connected = 1 if int(candidate.get("rel_count", 0) or 0) > 0 else 0
        return exact, connected, float(candidate.get("score", 0.0) or 0.0)

    return max(candidates, key=_rank)


def _build_entity_result(row: dict[str, Any]) -> GraphEntityResult:
    return GraphEntityResult(
        node_id=row.get("node_id"),
        name=str(row.get("name", "")),
        entity_type=str(row.get("entity_type", "UNKNOWN")),
        canonical_name=row.get("canonical_name"),
        score=float(row["score"]) if row.get("score") is not None else None,
        hop_count=int(row["hop_count"]) if row.get("hop_count") is not None else None,
        relationship_types=sorted({str(rel) for rel in row.get("rel_types", []) if rel}),
        properties=row.get("properties") or {},
    )


def _merge_section_rows(rows: list[dict[str, Any]]) -> list[GraphEntityResult]:
    merged: dict[str, GraphEntityResult] = {}

    for row in rows:
        item = _build_entity_result(row)
        key = item.node_id or f"{item.entity_type}:{item.name}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue

        existing.relationship_types = sorted(
            set(existing.relationship_types) | set(item.relationship_types)
        )
        if item.hop_count is not None:
            if existing.hop_count is None:
                existing.hop_count = item.hop_count
            else:
                existing.hop_count = min(existing.hop_count, item.hop_count)
        if not existing.properties:
            existing.properties = item.properties

    return sorted(
        merged.values(),
        key=lambda item: (
            item.hop_count if item.hop_count is not None else 999,
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


async def resolve_root_entity(
    driver,
    registry: RegistryLike,
    profile: QueryProfileDefinition,
    request: QueryProfileSearchRequest,
) -> GraphEntityResult:
    profile_map = _profile_map(registry)
    root_types = _root_entity_types(profile, profile_map)

    alias_matches = await _run_neo4j_query(
        driver,
        _ALIAS_QUERY,
        query=request.query_text,
        root_types=root_types,
    )
    alias_of_matches = await _run_neo4j_query(
        driver,
        _ALIAS_OF_QUERY,
        query=request.query_text,
        root_types=root_types,
    )
    fulltext_matches = await _run_neo4j_query(
        driver,
        _FULLTEXT_QUERY,
        query=request.query_text,
        root_types=root_types,
    )

    all_matches = alias_matches + alias_of_matches + fulltext_matches
    candidate = _select_best_candidate(all_matches, request.query_text)
    if candidate is None:
        raise QueryRootNotFoundError(
            f"No matching root entity found for '{request.query_text}'"
        )

    # If the best candidate has no domain relationships, try co-extracted
    # siblings from ALL zero-rel candidates (different entity nodes for the
    # same name may be linked to different chunks).
    if int(candidate.get("rel_count", 0) or 0) == 0:
        all_candidates = all_matches
        zero_rel_ids = list(dict.fromkeys(
            c["node_id"] for c in all_candidates
            if c.get("node_id") and int(c.get("rel_count", 0) or 0) == 0
        ))
        for node_id in zero_rel_ids:
            fallback_rows = await _run_neo4j_query(
                driver,
                _COEXTRACTED_FALLBACK_QUERY,
                node_id=node_id,
                root_types=root_types,
            )
            if fallback_rows:
                candidate = fallback_rows[0]
                break

    resolved = _build_entity_result(candidate)
    if request.include_aliases and resolved.node_id:
        alias_rows = await _run_neo4j_query(
            driver,
            _ALIASES_BY_NODE_ID_QUERY,
            node_id=resolved.node_id,
        )
        if alias_rows:
            resolved.aliases = sorted(
                {str(alias) for alias in alias_rows[0].get("aliases", []) if alias}
            )
    return resolved


def _rel_spec(step: QueryProfileStep) -> str:
    rel_types = []
    for rel_type in step.rel_types:
        if not _REL_TYPE_RE.match(rel_type):
            raise ValueError(f"Invalid relationship type '{rel_type}' in query profile")
        rel_types.append(rel_type)
    return f"[:{'|'.join(rel_types)}*{step.min_hops}..{step.max_hops}]"


def _step_pattern(start_alias: str, end_alias: str, step: QueryProfileStep) -> str:
    rel_spec = _rel_spec(step)
    if step.direction == "out":
        return f"({start_alias})-{rel_spec}->({end_alias}:Entity)"
    return f"({start_alias})<-{rel_spec}-({end_alias}:Entity)"


def _compile_traversal_arm(traversal: QueryProfileTraversal) -> str:
    lines = []
    rel_exprs: list[str] = []
    hop_exprs: list[str] = []
    current_alias = "root"

    for idx, step in enumerate(traversal.steps, start=1):
        path_alias = f"p{idx}"
        end_alias = "n" if idx == len(traversal.steps) else f"n{idx}"
        lines.append(f"MATCH {path_alias} = {_step_pattern(current_alias, end_alias, step)}")
        rel_exprs.append(f"[rel IN relationships({path_alias}) | type(rel)]")
        hop_exprs.append(f"length({path_alias})")
        current_alias = end_alias

    rel_join = " + ".join(rel_exprs) if rel_exprs else "[]"
    hop_join = " + ".join(hop_exprs) if hop_exprs else "0"
    lines.append(
        f"RETURN n, {rel_join} AS rel_types, {hop_join} AS hop_count"
    )
    return "\n".join(lines)


def _compile_section_query(profile: QueryProfileDefinition) -> str:
    if profile.kind != "section":
        raise ValueError(f"Profile '{profile.id}' is not a section profile")
    if not profile.traversals:
        raise ValueError(f"Profile '{profile.id}' has no traversals")

    arms = "\n        UNION\n".join(
        _compile_traversal_arm(traversal) for traversal in profile.traversals
    )
    return f"""
    MATCH (root:Entity {{id: $root_id}})
    CALL (root) {{
        {arms}
    }}
    WITH n, rel_types, hop_count
    WHERE n.id IS NOT NULL
      AND n.id <> $root_id
      AND (size($target_entity_types) = 0 OR n.entity_type IN $target_entity_types)
    RETURN n.id AS node_id,
           n.name AS name,
           n.canonical_name AS canonical_name,
           n.entity_type AS entity_type,
           properties(n) AS properties,
           rel_types,
           hop_count
    LIMIT $limit
    """


async def _fetch_section_items(
    driver,
    resolved: GraphEntityResult,
    request: QueryProfileSearchRequest,
    profile: QueryProfileDefinition,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []

    query = _compile_section_query(profile)
    rows = await _run_neo4j_query(
        driver,
        query,
        root_id=resolved.node_id,
        target_entity_types=profile.target_entity_types,
        limit=request.top_k,
    )
    return _merge_section_rows(rows)


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
               tc.chunk_text
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
               ic.chunk_text
        FROM retrieval.image_chunks ic
        JOIN ingest.documents d ON d.id = ic.document_id
        WHERE ic.id::text = ANY(:ids)
        """
    )
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    evidence_map: dict[str, GraphEvidenceItem] = {}
    for row in rows:
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
        )
    return evidence_map


async def attach_evidence(
    driver,
    db: AsyncSession,
    items: list[GraphEntityResult],
    limit: int,
) -> None:
    entity_ids = [item.node_id for item in items if item.node_id]
    if not entity_ids:
        return

    rows = await _run_neo4j_query(
        driver,
        _EVIDENCE_REFS_QUERY,
        entity_ids=entity_ids,
        limit=limit,
    )

    refs_by_entity: dict[str, list[dict[str, str]]] = {}
    chunk_ids: list[str] = []
    for row in rows:
        entity_id = str(row.get("entity_id", ""))
        refs = row.get("refs") or []
        refs_by_entity[entity_id] = refs
        for ref in refs:
            chunk_id = ref.get("chunk_id")
            if chunk_id:
                chunk_ids.append(str(chunk_id))

    chunk_map = await _fetch_chunk_evidence(db, list(dict.fromkeys(chunk_ids)))

    for item in items:
        if not item.node_id:
            continue
        refs = refs_by_entity.get(item.node_id, [])
        item.evidence = [
            chunk_map[str(ref["chunk_id"])]
            for ref in refs
            if ref.get("chunk_id") and str(ref["chunk_id"]) in chunk_map
        ]


async def execute_section_search(
    driver,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    registry: RegistryLike | None = None,
) -> QueryProfileSectionResponse:
    registry = registry or await get_required_active_registry(db)
    profile = _get_profile(registry, request.profile_id)
    if profile.kind != "section":
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a section query profile"
        )

    resolved = await resolve_root_entity(driver, registry, profile, request)
    items = await _fetch_section_items(driver, resolved, request, profile)

    if request.include_evidence:
        await attach_evidence(driver, db, [resolved] + items, request.evidence_top_k)

    return QueryProfileSectionResponse(
        registry_id=getattr(registry, "id", None),
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        items=items,
        total=len(items),
    )


async def execute_dossier_search(
    driver,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    registry: RegistryLike | None = None,
) -> QueryProfileDossierResponse:
    registry = registry or await get_required_active_registry(db)
    profile_map = _profile_map(registry)
    profile = _get_profile(registry, request.profile_id)
    if profile.kind != "dossier":
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a dossier query profile"
        )

    resolved = await resolve_root_entity(driver, registry, profile, request)

    sections: list[QueryProfileDossierSection] = []
    all_items: list[GraphEntityResult] = [resolved]
    for section_id in profile.section_profile_ids:
        section_profile = profile_map.get(section_id)
        if section_profile is None or section_profile.kind != "section":
            continue
        items = await _fetch_section_items(driver, resolved, request, section_profile)
        all_items.extend(items)
        sections.append(
            QueryProfileDossierSection(
                profile_id=section_profile.id,
                profile_label=section_profile.label,
                items=items,
                total=len(items),
            )
        )

    if request.include_evidence:
        await attach_evidence(driver, db, all_items, request.evidence_top_k)

    return QueryProfileDossierResponse(
        registry_id=getattr(registry, "id", None),
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        aliases=resolved.aliases if request.include_aliases else [],
        sections=sections,
    )
