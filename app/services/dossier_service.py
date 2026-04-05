"""Deterministic dossier queries for system-centric exact retrieval via GraphStore.

Uses fixed GraphStore method calls with parameter binding only -- no dynamic
query generation from natural language.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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

# Relationship types used for section traversals
STRUCTURE_REL_TYPES = [
    "HAS_SUBSYSTEM", "HAS_COMPONENT", "HAS_STAGE", "PART_OF",
]

RF_REL_TYPES = [
    "OPERATES_IN_BAND", "USES_WAVEFORM", "USES_MODULATION", "EMITS",
    "RADIATES", "RECEIVES", "HAS_SIGNATURE", "HAS_SCAN", "HAS_ANTENNA",
    "HAS_TRANSMITTER", "HAS_RECEIVER", "HAS_PROCESSING_CHAIN", "HAS_SEEKER",
    "SPECIFIED_BY",
]

PERFORMANCE_REL_TYPES = [
    "HAS_PERFORMANCE", "PROVIDES", "SPECIFIED_BY", "TRACKS", "GUIDES",
    "DETECTS", "ENGAGES", "CUES", "DESIGNATES", "SUPPORTS_ENGAGEMENT_OF",
    "HAS_GUIDANCE", "HAS_TIMELINE",
]

ORG_PLATFORM_REL_TYPES = [
    "OPERATED_BY", "MANUFACTURED_BY", "INSTALLED_ON", "DEPLOYED_ON",
    "ASSOCIATED_WITH", "INTERFACES_WITH",
]


class SystemNotFoundError(LookupError):
    """Raised when a requested system cannot be resolved."""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _build_entity_result(
    entity: Any,
    *,
    score: float | None = None,
    hop_count: int | None = None,
    rel_types: list[str] | None = None,
) -> GraphEntityResult:
    """Convert a GraphStore GraphEntityResult to a schema GraphEntityResult."""
    from app.services.graph_store import GraphEntityResult as StoreEntity
    if isinstance(entity, StoreEntity):
        return GraphEntityResult(
            node_id=entity.node_id,
            name=entity.name,
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            score=score,
            hop_count=hop_count,
            relationship_types=sorted(set(rel_types or [])),
            properties=entity.properties,
        )
    # dict fallback
    return GraphEntityResult(
        node_id=entity.get("node_id", ""),
        name=entity.get("name", ""),
        entity_type=entity.get("entity_type", "UNKNOWN"),
        canonical_name=entity.get("canonical_name"),
        score=score,
        hop_count=hop_count,
        relationship_types=sorted(set(rel_types or [])),
        properties=entity.get("properties", {}),
    )


def _select_best_candidate(
    candidates: list[Any],
    requested_name: str,
) -> Any | None:
    if not candidates:
        return None

    wanted = _normalize(requested_name)

    def _rank(c: Any) -> tuple[int, float]:
        name = _normalize(getattr(c, "name", "") or "")
        canonical = _normalize(getattr(c, "canonical_name", "") or "")
        exact = 1 if wanted in {name, canonical} else 0
        return exact, 0.0

    return max(candidates, key=_rank)


async def resolve_system(
    graph_store: Any,
    request: SystemQueryRequest,
) -> GraphEntityResult:
    """Resolve a system name to the best matching root entity."""
    # 1. Try alias search
    alias_matches = await graph_store.search_by_alias(
        request.system_name,
    )
    alias_filtered = [
        m for m in alias_matches
        if getattr(m, "entity_type", "") in ROOT_ENTITY_TYPES
    ]

    # 2. Try fulltext search
    fulltext_matches = await graph_store.fulltext_search(
        request.system_name,
        entity_types=ROOT_ENTITY_TYPES,
        limit=10,
    )

    all_matches = alias_filtered + fulltext_matches
    candidate = _select_best_candidate(all_matches, request.system_name)
    if candidate is None:
        raise SystemNotFoundError(f"No matching system found for '{request.system_name}'")

    resolved = _build_entity_result(candidate, score=100.0)
    if request.include_aliases and resolved.node_id:
        # Aliases are already part of the graph — the frontend doesn't heavily depend on this
        resolved.aliases = []
    return resolved


async def _fetch_section(
    graph_store: Any,
    root_node_id: str,
    rel_types: list[str],
    target_entity_types: list[str],
    limit: int,
    depth: int = 2,
) -> list[GraphEntityResult]:
    """Fetch a section by traversing from root via specified relationship types."""
    neighbors = await graph_store.get_neighborhood(
        root_node_id,
        depth=depth,
        rel_types=rel_types,
    )
    # Filter to target entity types
    filtered = [
        n for n in neighbors
        if getattr(n, "entity_type", "") in target_entity_types
        or not target_entity_types
    ]
    return [
        _build_entity_result(n, hop_count=1)
        for n in filtered[:limit]
    ]


async def get_system_components(
    graph_store: Any,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        graph_store,
        resolved.node_id,
        rel_types=STRUCTURE_REL_TYPES,
        target_entity_types=[],  # all types
        limit=request.top_k,
        depth=3,
    )


async def get_system_rf_parameters(
    graph_store: Any,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        graph_store,
        resolved.node_id,
        rel_types=RF_REL_TYPES + STRUCTURE_REL_TYPES,
        target_entity_types=RF_ENTITY_TYPES,
        limit=request.top_k,
    )


async def get_system_performance(
    graph_store: Any,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        graph_store,
        resolved.node_id,
        rel_types=PERFORMANCE_REL_TYPES + STRUCTURE_REL_TYPES,
        target_entity_types=PERFORMANCE_ENTITY_TYPES,
        limit=request.top_k,
    )


async def get_system_organizations_platforms(
    graph_store: Any,
    resolved: GraphEntityResult,
    request: SystemQueryRequest,
) -> list[GraphEntityResult]:
    if not resolved.node_id:
        return []
    return await _fetch_section(
        graph_store,
        resolved.node_id,
        rel_types=ORG_PLATFORM_REL_TYPES,
        target_entity_types=ORG_PLATFORM_ENTITY_TYPES,
        limit=request.top_k,
    )


async def _fetch_chunk_evidence(
    db: AsyncSession,
    chunk_ids: list[str],
) -> dict[str, GraphEvidenceItem]:
    """Look up chunk content from Postgres for evidence display."""
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
    graph_store: Any,
    db: AsyncSession,
    items: list[GraphEntityResult],
    limit: int,
) -> None:
    """For each entity, look up EXTRACTED_FROM chunk refs and load evidence from Postgres."""
    all_chunk_ids: list[str] = []
    entity_chunk_map: dict[str, list[str]] = {}

    for item in items:
        if not item.node_id:
            continue
        try:
            linked = await graph_store.get_ontology_linked_chunks(item.node_id)
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


async def build_system_dossier(
    graph_store: Any,
    db: AsyncSession,
    request: SystemQueryRequest,
) -> SystemDossierResponse:
    resolved = await resolve_system(graph_store, request)

    components = await get_system_components(graph_store, resolved, request)
    rf_parameters = await get_system_rf_parameters(graph_store, resolved, request)
    performance = await get_system_performance(graph_store, resolved, request)
    organizations_platforms = await get_system_organizations_platforms(graph_store, resolved, request)

    if request.include_evidence:
        all_items = [resolved] + components + rf_parameters + performance + organizations_platforms
        await attach_evidence(graph_store, db, all_items, request.evidence_top_k)

    return SystemDossierResponse(
        resolved_system=resolved,
        aliases=resolved.aliases if request.include_aliases else [],
        components=components,
        rf_parameters=rf_parameters,
        performance_characteristics=performance,
        organizations_platforms=organizations_platforms,
    )


async def build_section_response(
    graph_store: Any,
    db: AsyncSession,
    request: SystemQueryRequest,
    section: str,
) -> SystemSectionResponse:
    resolved = await resolve_system(graph_store, request)

    loaders = {
        "components": get_system_components,
        "rf_parameters": get_system_rf_parameters,
        "performance": get_system_performance,
    }
    if section not in loaders:
        raise ValueError(f"Unknown dossier section: {section}")

    items = await loaders[section](graph_store, resolved, request)
    if request.include_evidence:
        await attach_evidence(graph_store, db, [resolved] + items, request.evidence_top_k)

    return SystemSectionResponse(
        resolved_system=resolved,
        items=items,
        total=len(items),
    )
