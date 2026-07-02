"""CRUD and search endpoints for standalone ontology-backed query profiles.

Query profiles are first-class ``governance.query_profiles`` rows keyed by the
stable string ``profile_key`` (the old ``QueryProfileDefinition.id``). There is
no registry layer, no stored ontology copy, and no active/exposed gate — the
live ontology is served separately by ``GET /v1/ontology`` straight from the
air_defense_v3 Pydantic SSoT.
"""

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session, get_graph_store
from app.schemas.query_profiles import (
    OntologyResponse,
    QueryProfileCreate,
    QueryProfileDefinition,
    QueryProfileDossierResponse,
    QueryProfileResponse,
    QueryProfileSearchRequest,
    QueryProfileSectionResponse,
    QueryProfileUpdate,
)
from app.services.ontology_service import get_live_ontology
from app.services.query_profiles import (
    QueryProfileNotFoundError,
    QueryProfileReferencedError,
    QueryRootNotFoundError,
    create_profile,
    delete_profile,
    execute_dossier_search,
    execute_section_search,
    get_profile,
    get_required_profile,
    list_profiles,
    update_profile,
)

router = APIRouter(tags=["query-profiles"])

_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


# ---------------------------------------------------------------------------
# Write-path validation
#
# The flat CRUD payloads (``QueryProfileCreate``/``QueryProfileUpdate``) keep
# the profile body loose (a JSONB ``definition`` blob) so the UI can round-trip
# it, but a profile must still be *well-formed* before it lands in the table.
# We reconstruct a ``QueryProfileDefinition`` from the write payload purely to
# run its ``validate_shape`` model-validator + the RADAR/MISSILE root-type
# frozenset check, and separately verify every dossier ``section_profile_id``
# resolves to an existing row. Enforcement lives in the route (Task 3 owns the
# API/schema layer) so it applies uniformly to create + update.
# ---------------------------------------------------------------------------


def _assert_valid_shape(
    *,
    profile_key: str,
    label: str,
    kind: str,
    description: Optional[str],
    root_entity_types: list[str],
    definition: dict[str, Any],
) -> QueryProfileDefinition:
    """Build + validate a ``QueryProfileDefinition`` from a write payload.

    The ``definition`` body fields (traversals, section_profile_ids,
    profile_sections, include_associated_systems, …) are spread first; the
    column-backed fields (id/label/kind/root_entity_types) then override so the
    row's authoritative values win. Extra keys (e.g. ``profile_subgroup``) are
    ignored by Pydantic. A bad shape raises ``HTTPException(422)``."""
    candidate: dict[str, Any] = {
        **(definition or {}),
        "id": profile_key,
        "label": label,
        "kind": kind,
        "root_entity_types": list(root_entity_types or []),
    }
    if description is not None:
        candidate["description"] = description
    try:
        return QueryProfileDefinition.model_validate(candidate)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid query profile shape: {exc.errors()}",
        ) from exc


async def _assert_dossier_refs_exist(
    db: AsyncSession,
    *,
    kind: str,
    definition: dict[str, Any],
    self_key: str,
) -> None:
    """For a dossier, every ``section_profile_id`` must resolve to an existing
    ``profile_key`` row (replaces the removed registry ``_validate_profile_references``).
    Missing refs → ``HTTPException(422)``."""
    if kind != "dossier":
        return
    section_ids = [
        s for s in ((definition or {}).get("section_profile_ids") or []) if s
    ]
    missing: list[str] = []
    for section_id in section_ids:
        if section_id == self_key:
            continue
        if await get_profile(db, section_id) is None:
            missing.append(section_id)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Dossier profiles can only reference existing profiles. "
                f"Missing section_profile_ids: {', '.join(sorted(set(missing)))}"
            ),
        )


@router.get("/ontology", response_model=OntologyResponse)
async def get_ontology() -> OntologyResponse:
    """Live ontology, served straight from the air_defense_v3 Pydantic
    SSoT (no stored/registry copy) — see app.services.ontology_service."""
    return OntologyResponse.model_validate(get_live_ontology())


# ---------------------------------------------------------------------------
# Flat profile CRUD — keyed by the stable string ``profile_key``.
# ---------------------------------------------------------------------------


@router.get("/query-profiles", response_model=list[QueryProfileResponse])
async def list_query_profiles(
    enabled_only: bool = False,
    db: AsyncSession = Depends(get_async_session),
) -> list[QueryProfileResponse]:
    """List all query profiles (``?enabled_only=true`` to hide disabled)."""
    profiles = await list_profiles(db, enabled_only=enabled_only)
    return [QueryProfileResponse.model_validate(p) for p in profiles]


@router.post(
    "/query-profiles",
    response_model=QueryProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_query_profile(
    body: QueryProfileCreate,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileResponse:
    """Create a query profile. ``profile_key`` must be unique and the profile
    must be well-formed (``validate_shape`` + dossier refs)."""
    _assert_valid_shape(
        profile_key=body.profile_key,
        label=body.label,
        kind=body.kind,
        description=body.description,
        root_entity_types=body.root_entity_types,
        definition=body.definition,
    )
    await _assert_dossier_refs_exist(
        db, kind=body.kind, definition=body.definition, self_key=body.profile_key
    )

    if await get_profile(db, body.profile_key) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Query profile '{body.profile_key}' already exists",
        )

    try:
        profile = await create_profile(
            db,
            profile_key=body.profile_key,
            label=body.label,
            kind=body.kind,
            description=body.description,
            root_entity_types=body.root_entity_types,
            definition=body.definition,
            source_id=body.source_id,
            enabled=body.enabled,
            created_by=_PLACEHOLDER_USER,
        )
    except IntegrityError as exc:
        # TOCTOU backstop: a concurrent insert raced past the pre-check and won
        # the unique ``profile_key`` constraint. Surface 409, not a raw 500.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Query profile '{body.profile_key}' already exists",
        ) from exc
    await db.refresh(profile)
    return QueryProfileResponse.model_validate(profile)


@router.get(
    "/query-profiles/{profile_key}",
    response_model=QueryProfileResponse,
)
async def get_query_profile(
    profile_key: str,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileResponse:
    try:
        profile = await get_required_profile(db, profile_key)
    except QueryProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return QueryProfileResponse.model_validate(profile)


@router.put(
    "/query-profiles/{profile_key}",
    response_model=QueryProfileResponse,
)
async def update_query_profile(
    profile_key: str,
    body: QueryProfileUpdate,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileResponse:
    """Partial update — only fields present in the request body are applied.
    The MERGED profile (existing row + delta) must still be well-formed, so a
    partial update that would invalidate the profile is rejected (422)."""
    updates = body.model_dump(exclude_unset=True)

    try:
        existing = await get_required_profile(db, profile_key)
    except QueryProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Validate the MERGED view (not just the delta): a field the request omits
    # keeps its current value; ``definition`` is replace-whole (matching the
    # service's ``dict(definition or {})`` semantics).
    merged_kind = updates.get("kind", existing.kind)
    merged_definition = updates.get("definition", existing.definition or {})
    _assert_valid_shape(
        profile_key=profile_key,
        label=updates.get("label", existing.label),
        kind=merged_kind,
        description=updates.get("description", existing.description),
        root_entity_types=updates.get(
            "root_entity_types", existing.root_entity_types or []
        ),
        definition=merged_definition,
    )
    await _assert_dossier_refs_exist(
        db, kind=merged_kind, definition=merged_definition, self_key=profile_key
    )

    try:
        profile = await update_profile(db, profile_key, **updates)
    except QueryProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await db.refresh(profile)
    return QueryProfileResponse.model_validate(profile)


@router.delete(
    "/query-profiles/{profile_key}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_query_profile(
    profile_key: str,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete a profile. A section profile still referenced by a dossier
    profile cannot be deleted (409)."""
    try:
        await delete_profile(db, profile_key)
    except QueryProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueryProfileReferencedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# Exact graph search — profile resolved by ``profile_key`` (search request's
# ``profile_id`` string), honoring the profile's ``source_id`` scope.
# ---------------------------------------------------------------------------


@router.post(
    "/query-profiles/search/section",
    response_model=QueryProfileSectionResponse,
)
async def search_query_profile_section(
    body: QueryProfileSearchRequest,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileSectionResponse:
    graph_store = get_graph_store()
    try:
        return await execute_section_search(graph_store, db, body)
    except (
        QueryProfileNotFoundError,
        QueryRootNotFoundError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/query-profiles/search/dossier",
    response_model=QueryProfileDossierResponse,
)
async def search_query_profile_dossier(
    body: QueryProfileSearchRequest,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileDossierResponse:
    graph_store = get_graph_store()
    try:
        return await execute_dossier_search(graph_store, db, body)
    except (
        QueryProfileNotFoundError,
        QueryRootNotFoundError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
