"""Deterministic Neo4j dossier queries for system-centric exact retrieval.

This service uses fixed Cypher templates with parameter binding only. It does
not generate Cypher from natural language.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.graph_store import (
    GraphEntityResult,
    GraphEvidenceItem,
    SystemDossierResponse,
    SystemQueryRequest,
    SystemSectionResponse,
)

ROOT_ENTITY_TYPES = [
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

RF_ENTITY_TYPES = [
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

PERFORMANCE_ENTITY_TYPES = [
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

ORG_PLATFORM_ENTITY_TYPES = [
    "ORGANIZATION",
    "PLATFORM",
    "FORCE_STRUCTURE",
    "ASSEMBLY",
]

SYSTEM_ALIAS_QUERY = """
    MATCH (node:Entity)-[:HAS_ALIAS]->(a:Alias)
    WHERE toLower(a.alias_name) = toLower($query)
      AND node.entity_type IN $root_types
    RETURN node.id AS node_id,
           node.name AS name,
           node.canonical_name AS canonical_name,
           node.entity_type AS entity_type,
           properties(node) AS properties,
           100.0 AS score
    LIMIT 10
"""

SYSTEM_FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
    YIELD node, score
    WHERE node.entity_type IN $root_types
    RETURN node.id AS node_id,
           node.name AS name,
           node.canonical_name AS canonical_name,
           node.entity_type AS entity_type,
           properties(node) AS properties,
           score
    LIMIT 10
"""

SYSTEM_ALIASES_QUERY = """
    MATCH (node:Entity {id: $node_id})
    OPTIONAL MATCH (node)-[:HAS_ALIAS]->(a:Alias)
    RETURN collect(DISTINCT a.alias_name) AS aliases
"""

SYSTEM_COMPONENTS_QUERY = """
    MATCH (root:Entity {id: $root_id})
    CALL {
        WITH root
        MATCH path = (root)-[:HAS_SUBSYSTEM|HAS_COMPONENT|HAS_STAGE*1..3]->(n:Entity)
        RETURN n, [rel IN relationships(path) | type(rel)] AS rel_types, length(path) AS hop_count
        UNION
        WITH root
        MATCH path = (root)<-[:PART_OF*1..3]-(n:Entity)
        RETURN n, [rel IN relationships(path) | type(rel)] AS rel_types, length(path) AS hop_count
    }
    WHERE n.id IS NOT NULL AND n.id <> $root_id
    RETURN n.id AS node_id,
           n.name AS name,
           n.canonical_name AS canonical_name,
           n.entity_type AS entity_type,
           properties(n) AS properties,
           rel_types,
           hop_count
    LIMIT $limit
"""

SYSTEM_RF_QUERY = """
    MATCH (root:Entity {id: $root_id})
    CALL {
        WITH root
        MATCH path = (root)-[
            :OPERATES_IN_BAND|:USES_WAVEFORM|:USES_MODULATION|:EMITS|:RADIATES|:RECEIVES|
            :HAS_SIGNATURE|:HAS_SCAN|:HAS_ANTENNA|:HAS_TRANSMITTER|:HAS_RECEIVER|
            :HAS_PROCESSING_CHAIN|:HAS_SEEKER|:SPECIFIED_BY*1..2
        ]->(n:Entity)
        RETURN n, [rel IN relationships(path) | type(rel)] AS rel_types, length(path) AS hop_count
        UNION
        WITH root
        MATCH (root)-[:HAS_SUBSYSTEM|HAS_COMPONENT|HAS_STAGE]->(part:Entity)
        MATCH path = (part)-[
            :OPERATES_IN_BAND|:USES_WAVEFORM|:USES_MODULATION|:EMITS|:RADIATES|:RECEIVES|
            :HAS_SIGNATURE|:HAS_SCAN|:HAS_ANTENNA|:HAS_TRANSMITTER|:HAS_RECEIVER|
            :HAS_PROCESSING_CHAIN|:HAS_SEEKER|:SPECIFIED_BY*1..2
        ]->(n:Entity)
        RETURN n, ['HAS_SUBSYSTEM'] + [rel IN relationships(path) | type(rel)] AS rel_types, 1 + length(path) AS hop_count
    }
    WHERE n.id IS NOT NULL
      AND n.id <> $root_id
      AND n.entity_type IN $rf_types
    RETURN n.id AS node_id,
           n.name AS name,
           n.canonical_name AS canonical_name,
           n.entity_type AS entity_type,
           properties(n) AS properties,
           rel_types,
           hop_count
    LIMIT $limit
"""

SYSTEM_PERFORMANCE_QUERY = """
    MATCH (root:Entity {id: $root_id})
    CALL {
        WITH root
        MATCH path = (root)-[
            :HAS_PERFORMANCE|:PROVIDES|:SPECIFIED_BY|:TRACKS|:GUIDES|:DETECTS|:ENGAGES|
            :CUES|:DESIGNATES|:SUPPORTS_ENGAGEMENT_OF|:HAS_GUIDANCE|:HAS_TIMELINE*1..2
        ]->(n:Entity)
        RETURN n, [rel IN relationships(path) | type(rel)] AS rel_types, length(path) AS hop_count
        UNION
        WITH root
        MATCH (root)-[:HAS_SUBSYSTEM|HAS_COMPONENT|HAS_STAGE]->(part:Entity)
        MATCH path = (part)-[
            :HAS_PERFORMANCE|:PROVIDES|:SPECIFIED_BY|:TRACKS|:GUIDES|:DETECTS|:ENGAGES|
            :CUES|:DESIGNATES|:SUPPORTS_ENGAGEMENT_OF|:HAS_GUIDANCE|:HAS_TIMELINE*1..2
        ]->(n:Entity)
        RETURN n, ['HAS_SUBSYSTEM'] + [rel IN relationships(path) | type(rel)] AS rel_types, 1 + length(path) AS hop_count
    }
    WHERE n.id IS NOT NULL
      AND n.id <> $root_id
      AND n.entity_type IN $performance_types
    RETURN n.id AS node_id,
           n.name AS name,
           n.canonical_name AS canonical_name,
           n.entity_type AS entity_type,
           properties(n) AS properties,
           rel_types,
           hop_count
    LIMIT $limit
"""

SYSTEM_ORG_PLATFORM_QUERY = """
    MATCH (root:Entity {id: $root_id})
    MATCH path = (root)-[
        :OPERATED_BY|:MANUFACTURED_BY|:INSTALLED_ON|:DEPLOYED_ON|:ASSOCIATED_WITH|:INTERFACES_WITH*1..2
    ]->(n:Entity)
    WHERE n.id IS NOT NULL
      AND n.id <> $root_id
      AND n.entity_type IN $org_platform_types
    RETURN n.id AS node_id,
           n.name AS name,
           n.canonical_name AS canonical_name,
           n.entity_type AS entity_type,
           properties(n) AS properties,
           [rel IN relationships(path) | type(rel)] AS rel_types,
           length(path) AS hop_count
    LIMIT $limit
"""

EVIDENCE_REFS_QUERY = """
    UNWIND $entity_ids AS entity_id
    MATCH (n:Entity {id: entity_id})
    OPTIONAL MATCH (n)-[:EXTRACTED_FROM]->(c:ChunkRef)
    WITH entity_id,
         [ref IN collect({chunk_id: c.chunk_id, chunk_type: c.chunk_type}) WHERE ref.chunk_id IS NOT NULL][..$limit] AS refs
    RETURN entity_id, refs
"""


class SystemNotFoundError(LookupError):
    """Raised when a requested system cannot be resolved in Neo4j."""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


async def _run_neo4j_query(driver, query: str, **params) -> list[dict[str, Any]]:
    async with driver.session() as session:
        result = await session.run(query, **params)
        return await result.data()


def _select_best_candidate(
    candidates: list[dict[str, Any]],
    requested_name: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    wanted = _normalize(requested_name)

    def _rank(candidate: dict[str, Any]) -> tuple[int, float]:
        name = _normalize(str(candidate.get("name", "")))
        canonical = _normalize(str(candidate.get("canonical_name", "") or ""))
        exact = 1 if wanted in {name, canonical} else 0
        return exact, float(candidate.get("score", 0.0) or 0.0)

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
        existing.properties = existing.properties or item.properties

    return sorted(
        merged.values(),
        key=lambda item: (
            item.hop_count if item.hop_count is not None else 999,
            item.entity_type,
            item.name.casefold(),
        ),
    )


async def resolve_system(
    driver,
    request: SystemQueryRequest,
) -> GraphEntityResult:
    alias_matches = await _run_neo4j_query(
        driver,
        SYSTEM_ALIAS_QUERY,
        query=request.system_name,
        root_types=ROOT_ENTITY_TYPES,
    )
    fulltext_matches = await _run_neo4j_query(
        driver,
        SYSTEM_FULLTEXT_QUERY,
        query=request.system_name,
        root_types=ROOT_ENTITY_TYPES,
    )

    candidate = _select_best_candidate(alias_matches + fulltext_matches, request.system_name)
    if candidate is None:
        raise SystemNotFoundError(f"No matching system found for '{request.system_name}'")

    resolved = _build_entity_result(candidate)
    if request.include_aliases and resolved.node_id:
        alias_rows = await _run_neo4j_query(
            driver,
            SYSTEM_ALIASES_QUERY,
            node_id=resolved.node_id,
        )
        if alias_rows:
            resolved.aliases = sorted(
                {str(alias) for alias in alias_rows[0].get("aliases", []) if alias}
            )
    return resolved


async def _fetch_section(
    driver,
    query: str,
    *,
    root_id: str,
    limit: int,
    **params,
) -> list[GraphEntityResult]:
    rows = await _run_neo4j_query(
        driver,
        query,
        root_id=root_id,
        limit=limit,
        **params,
    )
    return _merge_section_rows(rows)


async def get_system_components(
    driver,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        driver,
        SYSTEM_COMPONENTS_QUERY,
        root_id=resolved.node_id,
        limit=request.top_k,
    )


async def get_system_rf_parameters(
    driver,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        driver,
        SYSTEM_RF_QUERY,
        root_id=resolved.node_id,
        limit=request.top_k,
        rf_types=RF_ENTITY_TYPES,
    )


async def get_system_performance(
    driver,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        driver,
        SYSTEM_PERFORMANCE_QUERY,
        root_id=resolved.node_id,
        limit=request.top_k,
        performance_types=PERFORMANCE_ENTITY_TYPES,
    )


async def get_system_organizations_platforms(
    driver,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        driver,
        SYSTEM_ORG_PLATFORM_QUERY,
        root_id=resolved.node_id,
        limit=request.top_k,
        org_platform_types=ORG_PLATFORM_ENTITY_TYPES,
    )


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
        EVIDENCE_REFS_QUERY,
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


async def build_system_dossier(
    driver,
    db: AsyncSession,
    request: SystemQueryRequest,
) -> SystemDossierResponse:
    resolved = await resolve_system(driver, request)

    components = await get_system_components(driver, resolved, request)
    rf_parameters = await get_system_rf_parameters(driver, resolved, request)
    performance = await get_system_performance(driver, resolved, request)
    organizations_platforms = await get_system_organizations_platforms(driver, resolved, request)

    if request.include_evidence:
        all_items = [resolved] + components + rf_parameters + performance + organizations_platforms
        await attach_evidence(driver, db, all_items, request.evidence_top_k)

    return SystemDossierResponse(
        resolved_system=resolved,
        aliases=resolved.aliases if request.include_aliases else [],
        components=components,
        rf_parameters=rf_parameters,
        performance_characteristics=performance,
        organizations_platforms=organizations_platforms,
    )


async def build_section_response(
    driver,
    db: AsyncSession,
    request: SystemQueryRequest,
    section: str,
) -> SystemSectionResponse:
    resolved = await resolve_system(driver, request)

    loaders = {
        "components": get_system_components,
        "rf_parameters": get_system_rf_parameters,
        "performance": get_system_performance,
    }
    if section not in loaders:
        raise ValueError(f"Unknown dossier section: {section}")

    items = await loaders[section](driver, resolved, request)
    if request.include_evidence:
        await attach_evidence(driver, db, [resolved] + items, request.evidence_top_k)

    return SystemSectionResponse(
        resolved_system=resolved,
        items=items,
        total=len(items),
    )
