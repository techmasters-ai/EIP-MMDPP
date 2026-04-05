"""Integration tests for the query profile CRUD and search endpoints.

All database and GraphStore interactions are mocked so that tests run without
external infrastructure.

Endpoints under test:
  GET    /v1/query-profiles/registries
  POST   /v1/query-profiles/registries
  GET    /v1/query-profiles/registries/{id}
  PUT    /v1/query-profiles/registries/{id}
  POST   /v1/query-profiles/registries/{id}/activate
  POST   /v1/query-profiles/registries/{id}/profiles
  PUT    /v1/query-profiles/registries/{id}/profiles/{profile_id}
  DELETE /v1/query-profiles/registries/{id}/profiles/{profile_id}
  GET    /v1/query-profiles
  GET    /v1/query-profiles/default-template
  POST   /v1/query-profiles/search/section
  POST   /v1/query-profiles/search/dossier
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")
_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_ONTOLOGY_DEFINITION = {
    "version": "1.0.0",
    "entity_types": [
        {"name": "EQUIPMENT_SYSTEM"},
        {"name": "COMPONENT"},
    ],
    "relationship_types": [
        {"name": "HAS_SUBSYSTEM"},
        {"name": "HAS_COMPONENT"},
    ],
}

_SECTION_PROFILE_DICT = {
    "id": "system_components",
    "label": "System Components",
    "kind": "section",
    "exposed": True,
    "root_entity_types": ["EQUIPMENT_SYSTEM"],
    "target_entity_types": ["COMPONENT"],
    "traversals": [
        {
            "steps": [
                {
                    "direction": "out",
                    "rel_types": ["HAS_SUBSYSTEM"],
                    "min_hops": 1,
                    "max_hops": 2,
                }
            ]
        }
    ],
    "section_profile_ids": [],
}

_DOSSIER_PROFILE_DICT = {
    "id": "system_dossier",
    "label": "System Dossier",
    "kind": "dossier",
    "exposed": True,
    "root_entity_types": ["EQUIPMENT_SYSTEM"],
    "target_entity_types": [],
    "traversals": [],
    "section_profile_ids": ["system_components"],
}


def _make_registry_row(
    *,
    registry_id: uuid.UUID | None = None,
    name: str = "Test Registry",
    description: str | None = None,
    ontology_name: str | None = "Test Ontology",
    ontology_version: str | None = "1.0.0",
    ontology_definition: dict | None = None,
    profiles: list | None = None,
    is_active: bool = False,
) -> MagicMock:
    """Build a MagicMock that looks like a QueryProfileRegistry ORM instance."""
    row = MagicMock()
    row.id = registry_id or uuid.uuid4()
    row.name = name
    row.description = description
    row.source_id = None
    row.ontology_name = ontology_name
    row.ontology_version = ontology_version
    row.ontology_definition = ontology_definition if ontology_definition is not None else _ONTOLOGY_DEFINITION
    row.profiles = profiles if profiles is not None else []
    row.is_active = is_active
    row.created_by = _PLACEHOLDER_USER
    row.created_at = _NOW
    row.updated_at = _NOW
    return row


def _mock_scalars_all(rows: list) -> MagicMock:
    """Create the result wrapper returned by ``db.execute(...).scalars().all()``."""
    scalars = MagicMock()
    scalars.all.return_value = rows
    result = MagicMock()
    result.scalars.return_value = scalars
    return result


def _mock_scalar_one_or_none(row) -> MagicMock:
    """Create the result wrapper for ``db.execute(...).scalar_one_or_none()``."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    return result


def _build_mock_db() -> AsyncMock:
    """Create a fully mocked async database session."""
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value=_mock_scalars_all([]))
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


# ---------------------------------------------------------------------------
# Fixture: yields (AsyncClient, mock_db) as a single tuple so that both the
# HTTP client and the mock database reference the same object.  This avoids
# the pytest-asyncio issue where dependent async fixtures resolve to separate
# instances.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def api():
    """Yield ``(client, mock_db)`` — an async HTTP client whose DB dependency
    is overridden with a shared ``AsyncMock``."""
    from app.db.session import get_async_session
    from app.main import create_app

    mock_db = _build_mock_db()
    app = create_app()

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_async_session] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac, mock_db
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /v1/query-profiles/registries -- list registries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_registries_empty(api):
    """Returns an empty list when no registries exist."""
    client, mock_db = api
    mock_db.execute.return_value = _mock_scalars_all([])

    resp = await client.get("/v1/query-profiles/registries")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_registries_returns_items(api):
    """Returns a populated list when registries exist."""
    client, mock_db = api
    row = _make_registry_row(name="Alpha Registry")
    mock_db.execute.return_value = _mock_scalars_all([row])

    resp = await client.get("/v1/query-profiles/registries")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "Alpha Registry"


# ---------------------------------------------------------------------------
# POST /v1/query-profiles/registries -- create registry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_registry_returns_201(api):
    """Creating a registry with name and ontology returns 201."""
    client, mock_db = api
    created_id = uuid.uuid4()

    async def fake_refresh(obj):
        obj.id = created_id
        obj.created_by = _PLACEHOLDER_USER
        obj.created_at = _NOW
        obj.updated_at = _NOW

    mock_db.refresh.side_effect = fake_refresh

    resp = await client.post(
        "/v1/query-profiles/registries",
        json={
            "name": "New Registry",
            "ontology_name": "Test Ontology",
            "ontology_version": "1.0.0",
            "ontology_definition": _ONTOLOGY_DEFINITION,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "New Registry"
    assert data["is_active"] is False
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_create_registry_with_profiles_rejected(api):
    """Supplying profiles on create returns 400."""
    client, _mock_db = api
    resp = await client.post(
        "/v1/query-profiles/registries",
        json={
            "name": "Bad Registry",
            "profiles": [_SECTION_PROFILE_DICT],
        },
    )
    assert resp.status_code == 400
    assert "ontology" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /v1/query-profiles/registries/{id} -- get one
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_registry_not_found(api):
    """Returns 404 for a missing registry."""
    client, mock_db = api
    mock_db.get.return_value = None
    missing_id = uuid.uuid4()

    resp = await client.get(f"/v1/query-profiles/registries/{missing_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_registry_found(api):
    """Returns 200 and the registry when found."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(registry_id=reg_id, name="Found Registry")
    mock_db.get.return_value = row

    resp = await client.get(f"/v1/query-profiles/registries/{reg_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Found Registry"


# ---------------------------------------------------------------------------
# PUT /v1/query-profiles/registries/{id} -- update
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_registry_not_found(api):
    """Returns 404 when updating a missing registry."""
    client, mock_db = api
    mock_db.get.return_value = None
    missing_id = uuid.uuid4()

    resp = await client.put(
        f"/v1/query-profiles/registries/{missing_id}",
        json={"name": "Updated"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_registry_name(api):
    """Updating the name returns 200 with the new name."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(registry_id=reg_id, name="Original")
    mock_db.get.return_value = row

    resp = await client.put(
        f"/v1/query-profiles/registries/{reg_id}",
        json={"name": "Updated Name"},
    )
    assert resp.status_code == 200
    # The mock row's name should have been mutated by the endpoint
    assert row.name == "Updated Name"


# ---------------------------------------------------------------------------
# POST /v1/query-profiles/registries/{id}/activate -- activate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_activate_registry(api):
    """Activating an existing registry returns 200 with is_active=True."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(registry_id=reg_id, is_active=False)
    mock_db.get.return_value = row

    resp = await client.post(f"/v1/query-profiles/registries/{reg_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True


@pytest.mark.asyncio
async def test_activate_registry_not_found(api):
    """Activating a missing registry returns 404."""
    client, mock_db = api
    mock_db.get.return_value = None
    missing_id = uuid.uuid4()

    resp = await client.post(f"/v1/query-profiles/registries/{missing_id}/activate")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/query-profiles/registries/{id}/profiles -- append profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_profile_to_active_registry(api):
    """Appending a section profile to an active registry with ontology returns 201."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[],
    )
    mock_db.get.return_value = row

    resp = await client.post(
        f"/v1/query-profiles/registries/{reg_id}/profiles",
        json=_SECTION_PROFILE_DICT,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["profiles"]) == 1
    assert data["profiles"][0]["id"] == "system_components"


@pytest.mark.asyncio
async def test_append_duplicate_profile_409(api):
    """Appending a profile with an existing id returns 409."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[_SECTION_PROFILE_DICT],
    )
    mock_db.get.return_value = row

    resp = await client.post(
        f"/v1/query-profiles/registries/{reg_id}/profiles",
        json=_SECTION_PROFILE_DICT,
    )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_append_profile_inactive_registry_400(api):
    """Appending to an inactive registry returns 400."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=False,
        ontology_definition=_ONTOLOGY_DEFINITION,
    )
    mock_db.get.return_value = row

    resp = await client.post(
        f"/v1/query-profiles/registries/{reg_id}/profiles",
        json=_SECTION_PROFILE_DICT,
    )
    assert resp.status_code == 400
    assert "active" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_append_profile_no_ontology_400(api):
    """Appending when ontology_definition is empty returns 400."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition={},
    )
    mock_db.get.return_value = row

    resp = await client.post(
        f"/v1/query-profiles/registries/{reg_id}/profiles",
        json=_SECTION_PROFILE_DICT,
    )
    assert resp.status_code == 400
    assert "ontology" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# PUT /v1/query-profiles/registries/{id}/profiles/{profile_id} -- update profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_profile_id_mismatch_400(api):
    """Path profile_id must match body id."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[_SECTION_PROFILE_DICT],
    )
    mock_db.get.return_value = row

    body = {**_SECTION_PROFILE_DICT, "id": "different_id"}
    resp = await client.put(
        f"/v1/query-profiles/registries/{reg_id}/profiles/system_components",
        json=body,
    )
    assert resp.status_code == 400
    assert "match" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_update_profile_not_found_404(api):
    """Updating a profile that does not exist returns 404."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[_SECTION_PROFILE_DICT],
    )
    mock_db.get.return_value = row

    body = {**_SECTION_PROFILE_DICT, "id": "nonexistent"}
    resp = await client.put(
        f"/v1/query-profiles/registries/{reg_id}/profiles/nonexistent",
        json=body,
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/query-profiles/registries/{id}/profiles/{profile_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_profile_success(api):
    """Deleting an unreferenced section profile succeeds."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[_SECTION_PROFILE_DICT],
    )
    mock_db.get.return_value = row

    resp = await client.delete(
        f"/v1/query-profiles/registries/{reg_id}/profiles/system_components"
    )
    assert resp.status_code == 200
    assert len(resp.json()["profiles"]) == 0


@pytest.mark.asyncio
async def test_delete_profile_referenced_by_dossier_400(api):
    """Cannot delete a section profile that is referenced by a dossier profile."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[_SECTION_PROFILE_DICT, _DOSSIER_PROFILE_DICT],
    )
    mock_db.get.return_value = row

    resp = await client.delete(
        f"/v1/query-profiles/registries/{reg_id}/profiles/system_components"
    )
    assert resp.status_code == 400
    assert "referenced" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_profile_not_found_404(api):
    """Deleting a non-existent profile returns 404."""
    client, mock_db = api
    reg_id = uuid.uuid4()
    row = _make_registry_row(
        registry_id=reg_id,
        is_active=True,
        ontology_definition=_ONTOLOGY_DEFINITION,
        profiles=[],
    )
    mock_db.get.return_value = row

    resp = await client.delete(
        f"/v1/query-profiles/registries/{reg_id}/profiles/nonexistent"
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/query-profiles -- active profiles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_active_profiles_no_registry(api):
    """Returns empty exposed_profiles when no registry is active."""
    client, mock_db = api
    mock_db.execute.return_value = _mock_scalar_one_or_none(None)

    resp = await client.get("/v1/query-profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["registry"] is None
    assert data["exposed_profiles"] == []


@pytest.mark.asyncio
async def test_active_profiles_with_registry(api):
    """Returns exposed profiles from the active registry."""
    client, mock_db = api
    row = _make_registry_row(
        is_active=True,
        profiles=[_SECTION_PROFILE_DICT],
    )
    mock_db.execute.return_value = _mock_scalar_one_or_none(row)

    resp = await client.get("/v1/query-profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["registry"] is not None
    assert len(data["exposed_profiles"]) == 1
    assert data["exposed_profiles"][0]["id"] == "system_components"


# ---------------------------------------------------------------------------
# GET /v1/query-profiles/default-template -- template
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_template_returns_200(api):
    """The default template endpoint returns a valid registry create payload."""
    client, _mock_db = api
    resp = await client.get("/v1/query-profiles/default-template")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "profiles" in data
    assert isinstance(data["profiles"], list)


# ---------------------------------------------------------------------------
# POST /v1/query-profiles/search/section -- section search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_section_no_active_registry_404(api):
    """Section search returns 404 when there is no active registry."""
    client, mock_db = api
    mock_db.execute.return_value = _mock_scalar_one_or_none(None)

    with patch(
        "app.api.v1.query_profiles.get_graph_store"
    ) as mock_gs:
        mock_store = MagicMock()
        mock_gs.return_value = mock_store

        resp = await client.post(
            "/v1/query-profiles/search/section",
            json={
                "profile_id": "system_components",
                "query_text": "AN/MPQ-65",
            },
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_dossier_no_active_registry_404(api):
    """Dossier search returns 404 when there is no active registry."""
    client, mock_db = api
    mock_db.execute.return_value = _mock_scalar_one_or_none(None)

    with patch(
        "app.api.v1.query_profiles.get_graph_store"
    ) as mock_gs:
        mock_store = MagicMock()
        mock_gs.return_value = mock_store

        resp = await client.post(
            "/v1/query-profiles/search/dossier",
            json={
                "profile_id": "system_dossier",
                "query_text": "AN/MPQ-65",
            },
        )
    assert resp.status_code == 404
