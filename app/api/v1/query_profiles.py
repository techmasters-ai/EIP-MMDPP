"""CRUD and search endpoints for standalone ontology-backed query profiles.

Query profiles are first-class ``governance.query_profiles`` rows keyed by the
stable string ``profile_key`` (the old ``QueryProfileDefinition.id``). There is
no registry layer, no stored ontology copy, and no active/exposed gate — the
live ontology is served separately by ``GET /v1/ontology`` straight from the
air_defense_v3 Pydantic SSoT.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session, get_graph_store
from app.schemas.query_profiles import (
    OntologyResponse,
    QueryProfileCreate,
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
    get_required_profile,
    list_profiles,
    update_profile,
)

router = APIRouter(tags=["query-profiles"])

_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


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
    """Create a query profile. ``profile_key`` must be unique."""
    from app.services.query_profiles import get_profile

    if await get_profile(db, body.profile_key) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Query profile '{body.profile_key}' already exists",
        )

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
    """Partial update — only fields present in the request body are applied."""
    updates = body.model_dump(exclude_unset=True)
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
