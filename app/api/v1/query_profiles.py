"""CRUD and search endpoints for ontology-backed query profile registries."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session, get_graph_store
from app.models.query_profiles import QueryProfileRegistry
from app.schemas.query_profiles import (
    ActiveQueryProfilesResponse,
    QueryProfileDefinition,
    QueryProfileDossierResponse,
    QueryProfileRegistryCreate,
    QueryProfileRegistryResponse,
    QueryProfileRegistryUpdate,
    QueryProfileSearchRequest,
    QueryProfileSectionResponse,
)
from app.services.query_profiles import (
    QueryProfileNotFoundError,
    QueryProfileRegistryNotFoundError,
    QueryRootNotFoundError,
    active_registry_payload,
    build_default_registry_template,
    execute_dossier_search,
    execute_section_search,
    get_active_registry,
    registry_to_response,
)
from app.services.ontology_templates import invalidate_ontology_cache

router = APIRouter(tags=["query-profiles"])

_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _registry_profile_models(registry: QueryProfileRegistry) -> list[QueryProfileDefinition]:
    return [
        QueryProfileDefinition.model_validate(profile)
        for profile in (registry.profiles or [])
    ]


def _ensure_profile_editable(registry: QueryProfileRegistry) -> None:
    if not registry.is_active:
        raise HTTPException(
            status_code=400,
            detail=(
                "Query profiles can only be edited on an active ontology registry. "
                "Activate the ontology definition first."
            ),
        )
    if not isinstance(registry.ontology_definition, dict) or not registry.ontology_definition:
        raise HTTPException(
            status_code=400,
            detail=(
                "The active registry does not have an ontology definition yet. "
                "Save the ontology definition before creating query profiles."
            ),
        )


def _validate_profile_references(
    profile: QueryProfileDefinition,
    existing_profiles: list[QueryProfileDefinition],
    *,
    replacing_profile_id: str | None = None,
) -> None:
    if profile.kind != "dossier":
        return

    available_section_ids = {
        item.id
        for item in existing_profiles
        if item.kind == "section" and item.id != replacing_profile_id
    }
    missing = [
        section_id
        for section_id in profile.section_profile_ids
        if section_id not in available_section_ids
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dossier profiles can only reference existing section profiles. "
                f"Missing section ids: {', '.join(sorted(missing))}"
            ),
        )


@router.get(
    "/query-profiles/registries",
    response_model=list[QueryProfileRegistryResponse],
)
async def list_query_profile_registries(
    db: AsyncSession = Depends(get_async_session),
) -> list[QueryProfileRegistryResponse]:
    result = await db.execute(
        select(QueryProfileRegistry).order_by(
            QueryProfileRegistry.is_active.desc(),
            QueryProfileRegistry.updated_at.desc(),
        )
    )
    registries = result.scalars().all()
    return [registry_to_response(registry) for registry in registries]


@router.get(
    "/query-profiles/registries/{registry_id}",
    response_model=QueryProfileRegistryResponse,
)
async def get_query_profile_registry(
    registry_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")
    return registry_to_response(registry)


@router.post(
    "/query-profiles/registries",
    response_model=QueryProfileRegistryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_query_profile_registry(
    body: QueryProfileRegistryCreate,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    if body.profiles:
        raise HTTPException(
            status_code=400,
            detail=(
                "Create and activate an ontology registry first. "
                "Query profiles can only be added after an ontology definition is active."
            ),
        )

    if body.is_active:
        await db.execute(update(QueryProfileRegistry).values(is_active=False))

    registry = QueryProfileRegistry(
        name=body.name,
        description=body.description,
        source_id=body.source_id,
        ontology_name=body.ontology_name,
        ontology_version=body.ontology_version,
        ontology_definition=body.ontology_definition,
        profiles=[profile.model_dump(mode="json") for profile in body.profiles],
        is_active=body.is_active,
        created_by=_PLACEHOLDER_USER,
    )
    db.add(registry)
    await db.commit()
    await db.refresh(registry)
    invalidate_ontology_cache()
    return registry_to_response(registry)


@router.put(
    "/query-profiles/registries/{registry_id}",
    response_model=QueryProfileRegistryResponse,
)
async def update_query_profile_registry(
    registry_id: uuid.UUID,
    body: QueryProfileRegistryUpdate,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")

    if body.is_active is True:
        await db.execute(
            update(QueryProfileRegistry)
            .where(QueryProfileRegistry.id != registry_id)
            .values(is_active=False)
        )

    if body.profiles is not None:
        _ensure_profile_editable(registry)

    payload = body.model_dump(exclude_unset=True)
    for field in ("name", "description", "source_id", "ontology_name", "ontology_version", "ontology_definition", "is_active"):
        if field in payload:
            setattr(registry, field, payload[field])
    if body.profiles is not None:
        registry.profiles = [profile.model_dump(mode="json") for profile in body.profiles]

    await db.commit()
    await db.refresh(registry)
    invalidate_ontology_cache()
    return registry_to_response(registry)


@router.post(
    "/query-profiles/registries/{registry_id}/activate",
    response_model=QueryProfileRegistryResponse,
)
async def activate_query_profile_registry(
    registry_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")

    await db.execute(update(QueryProfileRegistry).values(is_active=False))
    registry.is_active = True
    await db.commit()
    await db.refresh(registry)
    invalidate_ontology_cache()
    return registry_to_response(registry)


@router.post(
    "/query-profiles/registries/{registry_id}/profiles",
    response_model=QueryProfileRegistryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def append_query_profile(
    registry_id: uuid.UUID,
    body: QueryProfileDefinition,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")

    _ensure_profile_editable(registry)

    profiles = _registry_profile_models(registry)
    if any(profile.id == body.id for profile in profiles):
        raise HTTPException(
            status_code=409,
            detail=f"Query profile '{body.id}' already exists in this registry",
        )

    _validate_profile_references(body, profiles)
    profiles.append(body)
    registry.profiles = [profile.model_dump(mode="json") for profile in profiles]
    await db.commit()
    await db.refresh(registry)
    return registry_to_response(registry)


@router.put(
    "/query-profiles/registries/{registry_id}/profiles/{profile_id}",
    response_model=QueryProfileRegistryResponse,
)
async def update_query_profile(
    registry_id: uuid.UUID,
    profile_id: str,
    body: QueryProfileDefinition,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")

    _ensure_profile_editable(registry)

    if body.id != profile_id:
        raise HTTPException(
            status_code=400,
            detail="Query profile id in the payload must match the URL path",
        )

    profiles = _registry_profile_models(registry)
    _validate_profile_references(body, profiles, replacing_profile_id=profile_id)
    updated = False
    next_profiles: list[QueryProfileDefinition] = []
    for profile in profiles:
        if profile.id == profile_id:
            next_profiles.append(body)
            updated = True
        else:
            next_profiles.append(profile)

    if not updated:
        raise HTTPException(status_code=404, detail="Query profile not found")

    registry.profiles = [profile.model_dump(mode="json") for profile in next_profiles]
    await db.commit()
    await db.refresh(registry)
    return registry_to_response(registry)


@router.delete(
    "/query-profiles/registries/{registry_id}/profiles/{profile_id}",
    response_model=QueryProfileRegistryResponse,
)
async def delete_query_profile(
    registry_id: uuid.UUID,
    profile_id: str,
    db: AsyncSession = Depends(get_async_session),
) -> QueryProfileRegistryResponse:
    registry = await db.get(QueryProfileRegistry, registry_id)
    if registry is None:
        raise HTTPException(status_code=404, detail="Query profile registry not found")

    _ensure_profile_editable(registry)

    profiles = _registry_profile_models(registry)
    referencing = [
        profile.id
        for profile in profiles
        if profile.kind == "dossier" and profile_id in profile.section_profile_ids
    ]
    if referencing:
        raise HTTPException(
            status_code=400,
            detail=(
                "This section profile is still referenced by dossier profiles: "
                + ", ".join(sorted(referencing))
            ),
        )
    next_profiles = [profile for profile in profiles if profile.id != profile_id]
    if len(next_profiles) == len(profiles):
        raise HTTPException(status_code=404, detail="Query profile not found")

    registry.profiles = [profile.model_dump(mode="json") for profile in next_profiles]
    await db.commit()
    await db.refresh(registry)
    return registry_to_response(registry)


@router.get("/query-profiles", response_model=ActiveQueryProfilesResponse)
async def get_active_query_profiles(
    db: AsyncSession = Depends(get_async_session),
) -> ActiveQueryProfilesResponse:
    registry = await get_active_registry(db)
    return active_registry_payload(registry)


@router.get(
    "/query-profiles/default-template",
    response_model=QueryProfileRegistryCreate,
)
async def get_default_query_profile_template() -> QueryProfileRegistryCreate:
    return build_default_registry_template()


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
        QueryProfileRegistryNotFoundError,
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
        QueryProfileRegistryNotFoundError,
        QueryProfileNotFoundError,
        QueryRootNotFoundError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
