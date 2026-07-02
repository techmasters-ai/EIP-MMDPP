"""Table-backed exact graph search using GraphStore traversal.

Query profiles are first-class ``governance.query_profiles`` rows
(``QueryProfile``) — there is no registry layer, no frozen ontology copy,
and no active/exposed gate. Each profile carries an optional Project-Source
scope (``source_id``); when set, root resolution + evidence + associated
systems are filtered to entities/chunks whose documents belong to that
source. ``source_id = None`` means Global (unfiltered).
"""

from __future__ import annotations

import types
import uuid
from typing import Any, Union, get_args, get_origin

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.query_profiles import QueryProfile
from app.schemas.graph_store import GraphEntityResult, GraphEvidenceItem
from app.schemas.query_profiles import (
    QueryProfileDossierResponse,
    QueryProfileDossierSection,
    QueryProfileSearchRequest,
    QueryProfileSectionResponse,
)
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
    edge_values: dict[str, Any] | None = None,
):
    """Walk canonical_cls.model_fields, pick fields whose
    json_schema_extra['profile_sections'] contains *profile_section*,
    group by 'profile_subgroup'. Skip fields where instance_data[name]
    is None. Returns deterministically ordered groups (by subgroup
    name asc; fields by name asc within group). Spec §4.3.

    The reserved section name ``"dossier"`` is a catch-all: rather than
    matching a literal ``profile_sections`` tag, it selects EVERY field
    that carries ANY profile_sections tag (i.e. every projectable field),
    still grouped by its ``profile_subgroup``. This yields the full
    cross-section entity dossier in one projection.

    Edge-typed fields (those whose ``json_schema_extra`` carries an
    ``edge_label``) are NOT vertex scalar properties, so they never appear
    in *instance_data*. Their value is supplied instead via *edge_values*
    (field_name -> projected target name(s)), pre-computed by
    :func:`_project_edge_fields` which traverses the edge from the resolved
    entity. When *edge_values* is absent (the default — e.g. legacy/unit
    callers), edge-typed fields resolve to None and are skipped exactly as
    before, so the scalar-only projection is byte-for-byte unchanged.
    """
    from app.schemas.query_profiles import (
        QueryProfileFieldEntry, QueryProfileFieldEvidence, QueryProfileFieldGroup,
    )

    groups_by_subgroup: dict[str, list[QueryProfileFieldEntry]] = {}

    # Phase 3 task 35: per-field evidence is persisted on the vertex as
    # _field_evidence: { field_name: [{chunk_id, snippet, element_uid, value}, ...] }
    # Read it (if present) and surface as QueryProfileFieldEvidence rows on
    # each entry. Old data lacking _field_evidence yields empty evidence —
    # the UI just renders no popover chip.
    field_evidence_blob: dict = instance_data.get("_field_evidence") or {}
    if not isinstance(field_evidence_blob, dict):
        field_evidence_blob = {}

    for fname, finfo in canonical_cls.model_fields.items():
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            continue
        sections = extra.get("profile_sections") or []
        if profile_section == "dossier":
            # Catch-all: any field carrying a profile tag is in scope.
            if not sections:
                continue
        elif profile_section not in sections:
            continue
        # Edge-typed fields are not vertex scalars. When edge projection has
        # run (*edge_values* is a dict, the real service path) the traversed
        # target name(s) are authoritative and an absent field means the edge
        # yielded nothing -> skip (no empty rows). When *edge_values* is None
        # (legacy/direct callers that never traversed) fall back to whatever
        # the caller placed in *instance_data* for that field.
        if extra.get("edge_label"):
            if edge_values is None:
                value = instance_data.get(fname)
            else:
                value = edge_values.get(fname)
        else:
            value = instance_data.get(fname)
        if value is None:
            continue
        subgroup = extra.get("profile_subgroup") or ""

        evidence_rows: list[QueryProfileFieldEvidence] = []
        for raw in field_evidence_blob.get(fname) or []:
            if not isinstance(raw, dict):
                continue
            evidence_rows.append(QueryProfileFieldEvidence(
                chunk_id=raw.get("chunk_id"),
                supporting_snippet=raw.get("snippet") or "",
                element_uid=raw.get("element_uid"),
            ))

        entry = QueryProfileFieldEntry(
            name=fname,
            label=_human_label(fname),
            value=value,
            description=finfo.description,
            examples=list(finfo.examples) if finfo.examples else None,
            enum=extra.get("enum"),
            evidence=evidence_rows,
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


def _edge_field_is_list(annotation: Any) -> bool:
    """True when *annotation* is (or wraps, through Optional/Union) a list.

    Determines the cardinality of an edge-typed field so its projected value
    is represented like a list-valued field (``list[str]``) vs a single-valued
    field (a scalar/joined string). Generalized — no per-field literals.
    """
    origin = get_origin(annotation)
    if origin is list:
        return True
    union_types: tuple[Any, ...] = (Union,)
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_types = (Union, union_type)
    if origin in union_types:
        return any(_edge_field_is_list(arg) for arg in get_args(annotation))
    return False


def _edge_field_specs(
    canonical_cls: type,
    requested_sections: list[str],
) -> list[tuple[str, str, str, bool]]:
    """Return ``(field_name, edge_label, direction, is_list)`` for every
    edge-typed profile field of *canonical_cls* that is in scope for
    *requested_sections*.

    A field is edge-typed when its ``json_schema_extra`` carries an
    ``edge_label``; it is projectable only if it ALSO carries a
    ``profile_sections`` tag. Section scoping mirrors
    :func:`_project_field_groups` exactly — the reserved ``"dossier"``
    catch-all selects every projectable edge field, otherwise the field's
    sections must intersect *requested_sections*. This keeps us from
    traversing (and paying a graph round-trip for) edge fields that could
    never be emitted for the requested sections. Direction defaults to
    ``"out"`` and may be overridden per field via
    ``json_schema_extra['edge_direction']``.
    """
    want_all = "dossier" in requested_sections
    wanted = set(requested_sections)
    specs: list[tuple[str, str, str, bool]] = []
    for fname, finfo in canonical_cls.model_fields.items():
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            continue
        edge_label = extra.get("edge_label")
        if not edge_label:
            continue
        sections = extra.get("profile_sections") or []
        if not sections:
            continue
        if not want_all and not (set(sections) & wanted):
            continue
        direction = extra.get("edge_direction") or "out"
        specs.append(
            (fname, str(edge_label), str(direction), _edge_field_is_list(finfo.annotation))
        )
    return specs


async def _project_edge_fields(
    canonical_cls: type,
    entity_rid: str,
    graph_store: Any,
    requested_sections: list[str],
    *,
    db: AsyncSession | None = None,
    source_id: uuid.UUID | None = None,
    doc_id_cache: dict[str, set[str]] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Project edge-typed profile fields by traversing each field's edge
    from *entity_rid* and collecting the linked target entities' names.

    Traverses the field's edge via ``graph_store.get_edge_neighbors`` — a
    native single-hop directed ArcadeDB MATCH over the field's ``edge_label``
    in the field's direction. (The section-kind ``get_directed_traversal`` is
    NOT reused here: its variable-depth ``while:`` clause mis-binds to the root
    for a single hop on this ArcadeDB build, so it cannot fetch the edge's
    target.) No raw SQL is issued here.

    When *source_id* is set, the traversed targets are filtered to that source
    the same way root/associated candidates are (via
    :func:`_filter_candidates_in_source`) so a scoped ``deployment``/dossier
    never leaks targets from other sources. Global (``source_id=None``) skips
    the filter entirely.

    Returns ``{field_name: value}`` where *value* is a ``list[str]`` for
    list-cardinality fields and a comma-joined string for single-cardinality
    fields. Fields whose traversal yields no in-scope target are omitted, so
    :func:`_project_field_groups` skips them exactly like a None scalar.
    """
    values: dict[str, Any] = {}
    if not entity_rid:
        return values
    for fname, edge_label, direction, is_list in _edge_field_specs(
        canonical_cls, requested_sections
    ):
        neighbors = await graph_store.get_edge_neighbors(
            entity_rid, edge_label, direction=direction, limit=limit,
        )
        # Defensively exclude the source entity itself. get_edge_neighbors
        # already excludes it in SQL, but keep the guard so no backend/quirk can
        # let an entity project its OWN name as the edge value.
        neighbors = [
            n for n in (neighbors or [])
            if _entity_node_id(n) != entity_rid
        ]
        if source_id is not None and db is not None and neighbors:
            neighbors = await _filter_candidates_in_source(
                neighbors, graph_store, db, source_id, doc_id_cache=doc_id_cache,
            )

        names: list[str] = []
        seen: set[str] = set()
        for neighbor in neighbors:
            if isinstance(neighbor, dict):
                raw_name = neighbor.get("name")
            else:
                raw_name = getattr(neighbor, "name", None)
            name = (raw_name or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        if not names:
            continue
        names.sort(key=str.casefold)
        values[fname] = names if is_list else ", ".join(names)
    return values


class QueryProfileNotFoundError(LookupError):
    """Raised when a profile is not available in the query_profiles table."""


class QueryProfileReferencedError(ValueError):
    """Raised when a section profile is still referenced by a dossier profile."""


class QueryRootNotFoundError(LookupError):
    """Raised when a requested root entity cannot be resolved."""


# ---------------------------------------------------------------------------
# Profile field accessors — duck-typed over both the QueryProfile SQLAlchemy
# model (flat columns + a JSONB ``definition`` blob) and the legacy
# QueryProfileDefinition Pydantic schema (all fields as attributes). This lets
# the executors accept either shape (existing unit tests still pass schema
# objects, live callers pass model rows).
# ---------------------------------------------------------------------------


def _p_kind(profile: Any) -> str:
    return getattr(profile, "kind", "") or ""


def _p_label(profile: Any) -> str:
    return getattr(profile, "label", "") or ""


def _p_key(profile: Any) -> str:
    """External string identifier: model ``profile_key`` or schema ``id``."""
    return getattr(profile, "profile_key", None) or getattr(profile, "id", "")


def _p_root_entity_types(profile: Any) -> list[str]:
    return list(getattr(profile, "root_entity_types", None) or [])


def _p_def_field(profile: Any, key: str, default: Any) -> Any:
    """Read a body field from the model's JSONB ``definition`` blob, falling
    back to a schema attribute of the same name."""
    if hasattr(profile, "definition"):
        defn = getattr(profile, "definition") or {}
        return defn.get(key, default)
    return getattr(profile, key, default)


def _p_traversals(profile: Any) -> list[Any]:
    return _p_def_field(profile, "traversals", []) or []


def _p_target_entity_types(profile: Any) -> list[str]:
    return list(_p_def_field(profile, "target_entity_types", []) or [])


def _p_profile_sections(profile: Any) -> list[str]:
    return list(_p_def_field(profile, "profile_sections", []) or [])


def _p_section_profile_ids(profile: Any) -> list[str]:
    return list(_p_def_field(profile, "section_profile_ids", []) or [])


def _p_include_associated(profile: Any) -> bool:
    return bool(_p_def_field(profile, "include_associated_systems", False))


def _p_enabled(profile: Any) -> bool:
    """Whether a profile is available for search. Duck-typed over the
    QueryProfile model (``enabled`` column), the schema object (``enabled``
    attribute), and a dict body — defaulting to True when absent."""
    if isinstance(profile, dict):
        return bool(profile.get("enabled", True))
    return bool(getattr(profile, "enabled", True))


def _step_field(step: Any, key: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def _traversal_steps(traversal: Any) -> list[Any]:
    if isinstance(traversal, dict):
        return traversal.get("steps") or []
    return getattr(traversal, "steps", None) or []


def _source_id_of(profile: Any) -> uuid.UUID | None:
    """The Project-Source scope of a profile (None = Global)."""
    return getattr(profile, "source_id", None)


# ---------------------------------------------------------------------------
# Table-backed CRUD
# ---------------------------------------------------------------------------

_UNSET: Any = object()


async def list_profiles(
    db: AsyncSession,
    enabled_only: bool = False,
) -> list[QueryProfile]:
    """Return all query profiles, ordered by ``profile_key`` for determinism."""
    stmt = select(QueryProfile)
    if enabled_only:
        stmt = stmt.where(QueryProfile.enabled.is_(True))
    stmt = stmt.order_by(QueryProfile.profile_key)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_profile(
    db: AsyncSession,
    profile_key: str,
) -> QueryProfile | None:
    """Load a single profile by its stable string ``profile_key``."""
    result = await db.execute(
        select(QueryProfile).where(QueryProfile.profile_key == profile_key)
    )
    return result.scalar_one_or_none()


async def get_required_profile(
    db: AsyncSession,
    profile_key: str,
) -> QueryProfile:
    profile = await get_profile(db, profile_key)
    if profile is None:
        raise QueryProfileNotFoundError(
            f"Query profile '{profile_key}' does not exist"
        )
    return profile


async def create_profile(
    db: AsyncSession,
    *,
    profile_key: str,
    label: str,
    kind: str,
    description: str | None = None,
    root_entity_types: list[str] | None = None,
    definition: dict[str, Any] | None = None,
    source_id: uuid.UUID | None = None,
    enabled: bool = True,
    created_by: uuid.UUID | None = None,
) -> QueryProfile:
    """Insert a new profile row and return it (transaction committed by caller)."""
    profile = QueryProfile(
        profile_key=profile_key,
        label=label,
        kind=kind,
        description=description,
        root_entity_types=list(root_entity_types or []),
        definition=dict(definition or {}),
        source_id=source_id,
        enabled=enabled,
        created_by=created_by,
    )
    db.add(profile)
    await db.flush()
    return profile


async def update_profile(
    db: AsyncSession,
    profile_key: str,
    *,
    label: Any = _UNSET,
    description: Any = _UNSET,
    kind: Any = _UNSET,
    root_entity_types: Any = _UNSET,
    definition: Any = _UNSET,
    source_id: Any = _UNSET,
    enabled: Any = _UNSET,
) -> QueryProfile:
    """Partial-update a profile by ``profile_key``; only provided fields change."""
    profile = await get_required_profile(db, profile_key)
    if label is not _UNSET:
        profile.label = label
    if description is not _UNSET:
        profile.description = description
    if kind is not _UNSET:
        profile.kind = kind
    if root_entity_types is not _UNSET:
        profile.root_entity_types = list(root_entity_types or [])
    if definition is not _UNSET:
        profile.definition = dict(definition or {})
    if source_id is not _UNSET:
        profile.source_id = source_id
    if enabled is not _UNSET:
        profile.enabled = enabled
    await db.flush()
    return profile


async def delete_profile(
    db: AsyncSession,
    profile_key: str,
) -> None:
    """Delete a profile by ``profile_key``.

    Preserves the dossier-referenced guard: a section profile cannot be
    deleted while another profile's ``definition['section_profile_ids']``
    references its ``profile_key``.
    """
    profile = await get_required_profile(db, profile_key)

    others = await list_profiles(db)
    referencing = sorted(
        other.profile_key
        for other in others
        if other.profile_key != profile_key
        and profile_key in _p_section_profile_ids(other)
    )
    if referencing:
        raise QueryProfileReferencedError(
            "This section profile is still referenced by dossier profiles: "
            + ", ".join(referencing)
        )

    await db.delete(profile)
    await db.flush()


# ---------------------------------------------------------------------------
# Root-type resolution + candidate selection
# ---------------------------------------------------------------------------


def _root_entity_types(
    profile: Any,
    section_profiles: list[Any] | None = None,
) -> list[str]:
    roots = _p_root_entity_types(profile)
    if roots:
        return roots
    if _p_kind(profile) != "dossier":
        return []

    merged: list[str] = []
    seen: set[str] = set()
    for section in section_profiles or []:
        for entity_type in _p_root_entity_types(section):
            if entity_type not in seen:
                seen.add(entity_type)
                merged.append(entity_type)
    return merged


def _collect_rel_types(profile: Any) -> list[str]:
    """Collect all relationship types from a profile's traversals."""
    rel_types: list[str] = []
    seen: set[str] = set()
    for traversal in _p_traversals(profile):
        for step in _traversal_steps(traversal):
            for rt in (_step_field(step, "rel_types") or []):
                if rt not in seen:
                    seen.add(rt)
                    rel_types.append(rt)
    return rel_types


def _max_depth(profile: Any) -> int:
    """Calculate max traversal depth from a profile's traversals."""
    max_d = 1
    for traversal in _p_traversals(profile):
        total = sum(
            _step_field(step, "max_hops", 1) for step in _traversal_steps(traversal)
        )
        max_d = max(max_d, total)
    return max_d


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


# ---------------------------------------------------------------------------
# Project-Source ("in-source") filtering
#
# in-source predicate for an entity =
#   it has ≥1 EXTRACTED_FROM chunk (ArcadeDB) whose document_id maps to an
#   ``ingest.documents`` row whose ``source_id`` == the profile's source_id.
#
# The graph gives us each entity's EXTRACTED_FROM chunk document_ids; Postgres
# (``ingest.documents``) is authoritative for document→source_id. We batch one
# Postgres lookup per candidate set (never per candidate).
# ---------------------------------------------------------------------------


def _entity_node_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("node_id", "") or "")
    return str(getattr(candidate, "node_id", "") or "")


async def _in_source_document_ids(
    db: AsyncSession,
    document_ids: set[str],
    source_id: uuid.UUID | None,
) -> set[str]:
    """Return the subset of *document_ids* whose ``ingest.documents`` row
    belongs to *source_id*. ``source_id=None`` is Global (all pass)."""
    ids = [d for d in document_ids if d]
    if not ids or source_id is None:
        return set(ids)
    sql = text(
        "SELECT id::text AS document_id FROM ingest.documents "
        "WHERE id::text = ANY(:ids) AND source_id = :source_id"
    )
    rows = (await db.execute(sql, {"ids": ids, "source_id": source_id})).fetchall()
    return {str(row[0]) for row in rows}


async def _entity_document_ids_batch(
    graph_store: Any,
    node_ids: list[str],
    *,
    cache: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """COMPLETE distinct EXTRACTED_FROM document_ids per node, in ONE graph
    call for the whole set (never per-candidate, never a chunk sample).

    Graph errors are NOT swallowed — a transient failure must surface as an
    error rather than silently emptying an entity's document set (which would
    wrongly drop the candidate under scope while it survives under Global).

    A caller-supplied *cache* memoizes per-node results across the candidate
    sets of a single resolve pass so the same entity is never re-queried.
    """
    cache = cache if cache is not None else {}
    result: dict[str, set[str]] = {}
    missing: list[str] = []
    for nid in node_ids:
        if not nid:
            result[nid] = set()
        elif nid in cache:
            result[nid] = cache[nid]
        else:
            missing.append(nid)

    if missing:
        fetched = await graph_store.get_entity_source_document_ids(
            list(dict.fromkeys(missing))
        )
        for nid in missing:
            docs = set(fetched.get(nid) or set())
            cache[nid] = docs
            result[nid] = docs

    return result


async def _filter_candidates_in_source(
    candidates: list[Any],
    graph_store: Any,
    db: AsyncSession,
    source_id: uuid.UUID | None,
    *,
    doc_id_cache: dict[str, set[str]] | None = None,
) -> list[Any]:
    """Keep only candidates that have an EXTRACTED_FROM document in *source_id*.

    No-op (returns *candidates* unchanged, no graph/DB round-trip) when
    *source_id* is None or the candidate set is empty — this preserves the
    Global path byte-for-byte. The in-source test uses each entity's COMPLETE
    document set, so membership never depends on an arbitrary chunk window.
    """
    if source_id is None or not candidates:
        return candidates

    node_ids = [_entity_node_id(c) for c in candidates]
    docs_by_node = await _entity_document_ids_batch(
        graph_store, node_ids, cache=doc_id_cache
    )
    all_doc_ids: set[str] = set()
    for docs in docs_by_node.values():
        all_doc_ids |= docs

    in_source = await _in_source_document_ids(db, all_doc_ids, source_id)
    return [
        candidate
        for candidate in candidates
        if docs_by_node.get(_entity_node_id(candidate), set()) & in_source
    ]


async def resolve_root_entity(
    graph_store: Any,
    profile: Any,
    request: QueryProfileSearchRequest,
    *,
    db: AsyncSession | None = None,
    source_id: uuid.UUID | None = None,
    section_profiles: list[Any] | None = None,
) -> GraphEntityResult:
    """Resolve the root entity using GraphStore alias + fulltext search.

    When *source_id* is set, EACH candidate set (alias, fulltext, direct,
    co-extracted) is filtered to in-source entities BEFORE selection — a
    globally best but out-of-source candidate must never mask a valid
    in-source one (review finding 3). ``source_id=None`` is unchanged.
    """
    root_types = _root_entity_types(profile, section_profiles)
    scoped = source_id is not None
    # Per-resolve memo so an entity appearing in multiple candidate sets is
    # queried for its document membership at most once.
    doc_id_cache: dict[str, set[str]] = {}

    # 1. Alias search
    alias_matches = await graph_store.search_by_alias(
        request.query_text,
    )
    alias_filtered = [
        m for m in alias_matches
        if not root_types or getattr(m, "entity_type", "") in root_types
    ]
    if scoped:
        alias_filtered = await _filter_candidates_in_source(
            alias_filtered, graph_store, db, source_id, doc_id_cache=doc_id_cache
        )

    # 2. Fulltext search
    fulltext_matches = await graph_store.fulltext_search(
        request.query_text,
        entity_types=root_types if root_types else None,
        limit=10,
    )
    if scoped:
        fulltext_matches = await _filter_candidates_in_source(
            fulltext_matches, graph_store, db, source_id, doc_id_cache=doc_id_cache
        )

    all_matches = alias_filtered + fulltext_matches

    # 3. Select best in-source candidate (filtering already applied above)
    candidate = _select_best_candidate(all_matches, request.query_text)
    if candidate is None:
        # 3b. Try direct name resolution
        resolved = await graph_store.resolve_root_entity(request.query_text)
        if resolved is not None and scoped:
            direct_in_source = await _filter_candidates_in_source(
                [resolved], graph_store, db, source_id, doc_id_cache=doc_id_cache
            )
            resolved = direct_in_source[0] if direct_in_source else None
        if resolved is not None:
            candidate = resolved
        else:
            # 4. Co-extracted fallback: find entities that co-occur with
            # a partial-match entity in the same source chunks. Only the
            # co-extracted fetch/selection is tolerant of failure; the
            # in-source filter is deliberately OUTSIDE the guard so its graph
            # errors propagate (consistent with alias/fulltext/direct) rather
            # than being swallowed into a spurious QueryRootNotFoundError.
            try:
                co_extracted = await graph_store.get_co_extracted_entities(
                    request.query_text, limit=5,
                )
                co_filtered = [
                    e for e in co_extracted
                    if not root_types or getattr(e, "entity_type", "") in root_types
                ]
            except Exception:
                co_filtered = []

            if scoped:
                co_filtered = await _filter_candidates_in_source(
                    co_filtered, graph_store, db, source_id,
                    doc_id_cache=doc_id_cache,
                )
            candidate = _select_best_candidate(co_filtered, request.query_text)

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
    profile: Any,
    *,
    db: AsyncSession | None = None,
    source_id: uuid.UUID | None = None,
    doc_id_cache: dict[str, set[str]] | None = None,
):
    """Fetch section items.

    Returns:
      - list[QueryProfileFieldGroup] when profile.kind == "section_properties"
      - list[GraphEntityResult]      when profile.kind == "section" (legacy)

    For section_properties profiles, resolves the root vertex's full property
    dict via get_entity_by_rid and projects it through _project_field_groups
    for each requested profile_section (spec §4.4). Edge-typed profile fields
    (e.g. ``deployment.platform`` = INSTALLED_ON) are not vertex scalars, so
    they are projected separately by traversing their edge from the resolved
    entity (:func:`_project_edge_fields`) and merged in. ``db`` + ``source_id``
    (both optional) enable the in-source filter for those traversed targets.

    For legacy section profiles, compiles the traversal steps into a native
    ArcadeDB MATCH pattern and returns matching neighbor entities.
    """
    if _p_kind(profile) == "section_properties":
        if not resolved.node_id:
            return []
        instance_data = await graph_store.get_entity_by_rid(resolved.node_id)
        canonical = _canonical_class_for(resolved.entity_type)
        sections = _p_profile_sections(profile)
        edge_values = await _project_edge_fields(
            canonical, resolved.node_id, graph_store, sections,
            db=db, source_id=source_id, doc_id_cache=doc_id_cache,
            limit=request.top_k,
        )
        groups: list = []
        for section in sections:
            groups.extend(
                _project_field_groups(
                    canonical, instance_data, section, edge_values=edge_values
                )
            )
        return groups

    if not resolved.node_id:
        return []

    # Run each traversal as a separate directed MATCH and merge the results.
    traversals_with_steps = [
        t for t in _p_traversals(profile) if _traversal_steps(t)
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
        target_types = _p_target_entity_types(profile)
        if target_types:
            neighbors = [n for n in neighbors if getattr(n, "entity_type", "") in target_types]
        neighbors = [n for n in neighbors if getattr(n, "node_id", "") != resolved.node_id]
        return _merge_section_results(neighbors[:request.top_k])

    neighbors: list = []
    for traversal in traversals_with_steps:
        steps = [
            {
                "direction": _step_field(step, "direction", "out"),
                "rel_types": _step_field(step, "rel_types", []),
                "min_hops": _step_field(step, "min_hops", 1),
                "max_hops": _step_field(step, "max_hops", 1),
            }
            for step in _traversal_steps(traversal)
        ]
        rows = await graph_store.get_directed_traversal(
            resolved.node_id,
            steps=steps,
            target_entity_types=_p_target_entity_types(profile) or None,
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
    source_id: uuid.UUID | None = None,
) -> dict[str, GraphEvidenceItem]:
    if not chunk_ids:
        return {}

    # When source-scoped, confine evidence chunks to documents belonging to
    # the profile's source — on BOTH UNION branches (review finding 4). The
    # query aliases ``ingest.documents`` as ``d``. With source_id=None the
    # predicate is absent and the SQL/params are byte-identical to before.
    source_clause = " AND d.source_id = :source_id" if source_id is not None else ""

    sql = text(
        f"""
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
        WHERE tc.id::text = ANY(:ids){source_clause}
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
        WHERE ic.id::text = ANY(:ids){source_clause}
        """
    )
    params: dict[str, Any] = {"ids": chunk_ids}
    if source_id is not None:
        params["source_id"] = source_id
    rows = (await db.execute(sql, params)).fetchall()
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
    *,
    source_id: uuid.UUID | None = None,
) -> None:
    """For each entity, look up EXTRACTED_FROM chunk refs and load evidence.

    When *source_id* is set, evidence is confined to that source's documents
    (review finding 4). ``source_id=None`` is unchanged.
    """
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

    chunk_map = await _fetch_chunk_evidence(
        db, list(dict.fromkeys(all_chunk_ids)), source_id=source_id
    )

    for item in items:
        if not item.node_id:
            continue
        chunk_ids = entity_chunk_map.get(item.node_id, [])
        item.evidence = [
            chunk_map[cid]
            for cid in chunk_ids
            if cid in chunk_map
        ]


async def _associated_systems(
    graph_store: Any,
    db: AsyncSession,
    resolved: GraphEntityResult,
    source_id: uuid.UUID | None,
) -> list[GraphEntityResult]:
    """Associated systems for the resolved root, source-filtered when scoped.

    When source-scoped, associated systems whose documents are not in the
    selected source are dropped (review finding 5) — not just their evidence.
    """
    if not resolved.node_id:
        return []
    related = await graph_store.get_associated_systems(resolved.node_id)
    if source_id is not None:
        related = await _filter_candidates_in_source(
            related, graph_store, db, source_id
        )
    return related


async def execute_section_search(
    graph_store: Any,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    profile: QueryProfile | None = None,
) -> QueryProfileSectionResponse:
    profile = profile or await get_required_profile(db, request.profile_id)
    if not _p_enabled(profile):
        raise QueryProfileNotFoundError(
            f"Query profile '{request.profile_id}' is not available (disabled)"
        )
    kind = _p_kind(profile)
    if kind not in ("section", "section_properties"):
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a section query profile"
        )

    source_id = _source_id_of(profile)
    # Per-search memo for entity→document membership, used by the section-item
    # source filter below (the resolve pass keeps its own internal memo).
    doc_id_cache: dict[str, set[str]] = {}
    resolved = await resolve_root_entity(
        graph_store, profile, request, db=db, source_id=source_id
    )
    raw = await _fetch_section_items(
        graph_store, resolved, request, profile,
        db=db, source_id=source_id, doc_id_cache=doc_id_cache,
    )

    if kind == "section_properties":
        field_groups = raw  # list[QueryProfileFieldGroup]
        related_systems: list[GraphEntityResult] = []
        if _p_include_associated(profile):
            related_systems = await _associated_systems(
                graph_store, db, resolved, source_id
            )
        if request.include_evidence and related_systems:
            await attach_evidence(
                graph_store, db, [resolved] + related_systems,
                request.evidence_top_k, source_id=source_id,
            )
        total = sum(len(g.fields) for g in field_groups) + len(related_systems)
        return QueryProfileSectionResponse(
            profile_id=_p_key(profile),
            profile_label=_p_label(profile),
            resolved_root=resolved,
            field_groups=field_groups,
            related_systems=related_systems,
            items=[],
            total=total,
        )

    items = raw  # list[GraphEntityResult]
    # When source-scoped, the traversal can return neighbor entities whose
    # documents belong to OTHER sources — drop them at the entity level before
    # they're returned/attached-evidence (Fix 1). Global (source_id=None) is a
    # no-op inside _filter_candidates_in_source, so this path is unchanged.
    if source_id is not None:
        items = await _filter_candidates_in_source(
            items, graph_store, db, source_id, doc_id_cache=doc_id_cache
        )
    if request.include_evidence:
        await attach_evidence(
            graph_store, db, [resolved] + items,
            request.evidence_top_k, source_id=source_id,
        )

    return QueryProfileSectionResponse(
        profile_id=_p_key(profile),
        profile_label=_p_label(profile),
        resolved_root=resolved,
        items=items,
        total=len(items),
    )


async def execute_dossier_search(
    graph_store: Any,
    db: AsyncSession,
    request: QueryProfileSearchRequest,
    *,
    profile: QueryProfile | None = None,
) -> QueryProfileDossierResponse:
    profile = profile or await get_required_profile(db, request.profile_id)
    if not _p_enabled(profile):
        raise QueryProfileNotFoundError(
            f"Query profile '{request.profile_id}' is not available (disabled)"
        )
    if _p_kind(profile) != "dossier":
        raise QueryProfileNotFoundError(
            f"Profile '{request.profile_id}' is not a dossier query profile"
        )

    source_id = _source_id_of(profile)
    # Per-search memo for entity→document membership, shared across every
    # section-kind item filter below (resolve keeps its own internal memo).
    doc_id_cache: dict[str, set[str]] = {}
    section_ids = _p_section_profile_ids(profile)

    # Load referenced section profiles from the table by profile_key.
    section_profiles: dict[str, QueryProfile] = {}
    for section_id in section_ids:
        section_profile = await get_profile(db, section_id)
        if section_profile is not None:
            section_profiles[section_id] = section_profile

    resolved = await resolve_root_entity(
        graph_store, profile, request,
        db=db, source_id=source_id,
        section_profiles=list(section_profiles.values()),
    )

    sections: list[QueryProfileDossierSection] = []
    all_items: list[GraphEntityResult] = [resolved]
    for section_id in section_ids:
        section_profile = section_profiles.get(section_id)
        if section_profile is None:
            continue
        if _p_kind(section_profile) not in ("section", "section_properties"):
            continue
        raw = await _fetch_section_items(
            graph_store, resolved, request, section_profile,
            db=db, source_id=source_id, doc_id_cache=doc_id_cache,
        )

        if _p_kind(section_profile) == "section_properties":
            field_groups = raw  # list[QueryProfileFieldGroup]
            related_systems: list[GraphEntityResult] = []
            if _p_include_associated(section_profile):
                related_systems = await _associated_systems(
                    graph_store, db, resolved, source_id
                )
            all_items.extend(related_systems)
            total = sum(len(g.fields) for g in field_groups) + len(related_systems)
            sections.append(
                QueryProfileDossierSection(
                    profile_id=_p_key(section_profile),
                    profile_label=_p_label(section_profile),
                    kind="section_properties",
                    field_groups=field_groups,
                    related_systems=related_systems,
                    items=[],
                    total=total,
                )
            )
        else:
            items = raw  # list[GraphEntityResult]
            # Source-filter traversal items at the entity level so a scoped
            # dossier section never returns entities from other sources (Fix 1).
            # Global (source_id=None) is a no-op → path unchanged.
            if source_id is not None:
                items = await _filter_candidates_in_source(
                    items, graph_store, db, source_id, doc_id_cache=doc_id_cache
                )
            all_items.extend(items)
            sections.append(
                QueryProfileDossierSection(
                    profile_id=_p_key(section_profile),
                    profile_label=_p_label(section_profile),
                    kind="section",
                    items=items,
                    total=len(items),
                )
            )

    if request.include_evidence:
        await attach_evidence(
            graph_store, db, all_items, request.evidence_top_k, source_id=source_id,
        )

    return QueryProfileDossierResponse(
        profile_id=_p_key(profile),
        profile_label=_p_label(profile),
        resolved_root=resolved,
        aliases=resolved.aliases if request.include_aliases else [],
        sections=sections,
    )
