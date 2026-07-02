"""Integration tests for the standalone query-profile CRUD and search endpoints.

The registry layer is gone: profiles are first-class ``governance.query_profiles``
rows keyed by the string ``profile_key``. All service + GraphStore interactions
are patched so the tests run without external infrastructure.

Endpoints under test:
  GET    /v1/ontology
  GET    /v1/query-profiles            (?enabled_only=bool)
  POST   /v1/query-profiles
  GET    /v1/query-profiles/{profile_key}
  PUT    /v1/query-profiles/{profile_key}
  DELETE /v1/query-profiles/{profile_key}
  POST   /v1/query-profiles/search/section
  POST   /v1/query-profiles/search/dossier

Plus assertions that the removed registry routes are gone.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")

_API = "app.api.v1.query_profiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile_row(
    *,
    profile_key: str = "system_components",
    label: str = "System Components",
    description: str | None = "A section profile",
    kind: str = "section_properties",
    root_entity_types: list[str] | None = None,
    definition: dict | None = None,
    source_id: uuid.UUID | None = None,
    enabled: bool = True,
) -> SimpleNamespace:
    """A duck-typed stand-in for a ``QueryProfile`` ORM row.

    ``QueryProfileResponse.model_validate`` reads attributes off it
    (``from_attributes=True``), so every field must hold a real typed value
    (a bare ``MagicMock`` would fail validation)."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        profile_key=profile_key,
        label=label,
        description=description,
        kind=kind,
        root_entity_types=root_entity_types
        if root_entity_types is not None
        else ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        definition=definition if definition is not None else {"profile_sections": ["components"]},
        source_id=source_id,
        enabled=enabled,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _build_mock_db() -> AsyncMock:
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.flush = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    return db


@pytest_asyncio.fixture
async def api():
    """Yield ``(client, mock_db)`` — an async HTTP client whose DB dependency
    is overridden with a shared ``AsyncMock``. Service functions are patched
    per-test, so the mock DB is only a placeholder the dependency yields."""
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
# GET /v1/ontology (Task 1 route preserved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ontology_route_still_served(api):
    client, _ = api
    resp = await client.get("/v1/ontology")
    assert resp.status_code == 200
    data = resp.json()
    names = {e["name"] for e in data["entity_types"]}
    assert "RADAR_SYSTEM" in names
    assert "MISSILE_SYSTEM" in names


# ---------------------------------------------------------------------------
# GET /v1/query-profiles -- list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_profiles_empty(api):
    client, _ = api
    with patch(f"{_API}.list_profiles", new=AsyncMock(return_value=[])):
        resp = await client.get("/v1/query-profiles")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_profiles_returns_items(api):
    client, _ = api
    rows = [
        _make_profile_row(profile_key="system_components"),
        _make_profile_row(profile_key="system_dossier", kind="dossier"),
    ]
    with patch(f"{_API}.list_profiles", new=AsyncMock(return_value=rows)):
        resp = await client.get("/v1/query-profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert [p["profile_key"] for p in data] == ["system_components", "system_dossier"]
    assert data[0]["source_id"] is None
    assert data[0]["enabled"] is True


@pytest.mark.asyncio
async def test_list_profiles_enabled_only_forwarded(api):
    client, _ = api
    mock = AsyncMock(return_value=[])
    with patch(f"{_API}.list_profiles", new=mock):
        resp = await client.get("/v1/query-profiles?enabled_only=true")
    assert resp.status_code == 200
    # enabled_only=True is threaded through to the service.
    assert mock.await_args.kwargs.get("enabled_only") is True


# ---------------------------------------------------------------------------
# POST /v1/query-profiles -- create
# ---------------------------------------------------------------------------


_VALID_SECTION_PROPERTIES = {
    "kind": "section_properties",
    "root_entity_types": ["RADAR_SYSTEM"],
    "definition": {"profile_sections": ["rf_parameters"]},
}


@pytest.mark.asyncio
async def test_create_profile_201(api):
    client, mock_db = api
    row = _make_profile_row(profile_key="new_profile", label="New Profile")
    create_mock = AsyncMock(return_value=row)
    with patch(f"{_API}.get_profile", new=AsyncMock(return_value=None)), patch(
        f"{_API}.create_profile", new=create_mock
    ):
        resp = await client.post(
            "/v1/query-profiles",
            json={
                "profile_key": "new_profile",
                "label": "New Profile",
                **_VALID_SECTION_PROPERTIES,
                "source_id": None,
                "enabled": True,
            },
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["profile_key"] == "new_profile"
    assert data["source_id"] is None
    # source_id + enabled are threaded into the service create call.
    kwargs = create_mock.await_args.kwargs
    assert kwargs["profile_key"] == "new_profile"
    assert kwargs["source_id"] is None
    assert kwargs["enabled"] is True
    assert kwargs["created_by"] == _PLACEHOLDER_USER


@pytest.mark.asyncio
async def test_create_profile_duplicate_409(api):
    client, _ = api
    existing = _make_profile_row(profile_key="dupe")
    with patch(f"{_API}.get_profile", new=AsyncMock(return_value=existing)):
        resp = await client.post(
            "/v1/query-profiles",
            json={"profile_key": "dupe", "label": "Dupe", **_VALID_SECTION_PROPERTIES},
        )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_profile_integrity_error_maps_409(api):
    """A TOCTOU race that slips past the pre-check and hits the unique
    ``profile_key`` constraint surfaces as 409, not an unhandled 500."""
    from sqlalchemy.exc import IntegrityError

    client, _ = api
    with patch(f"{_API}.get_profile", new=AsyncMock(return_value=None)), patch(
        f"{_API}.create_profile",
        new=AsyncMock(side_effect=IntegrityError("insert", {}, Exception("dup"))),
    ):
        resp = await client.post(
            "/v1/query-profiles",
            json={"profile_key": "racy", "label": "Racy", **_VALID_SECTION_PROPERTIES},
        )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_valid_dossier_201(api):
    """A dossier whose section_profile_ids all resolve is created."""
    client, _ = api
    row = _make_profile_row(profile_key="dossier_ok", kind="dossier")

    async def _lookup(_db, key):
        # referenced section exists; the dossier's own key does not (not a dup)
        return _make_profile_row(profile_key=key) if key == "system_components" else None

    with patch(f"{_API}.get_profile", new=AsyncMock(side_effect=_lookup)), patch(
        f"{_API}.create_profile", new=AsyncMock(return_value=row)
    ):
        resp = await client.post(
            "/v1/query-profiles",
            json={
                "profile_key": "dossier_ok",
                "label": "Dossier OK",
                "kind": "dossier",
                "root_entity_types": ["RADAR_SYSTEM"],
                "definition": {"section_profile_ids": ["system_components"]},
            },
        )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Write-path shape validation → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_section_without_traversals_422(api):
    """kind=section with no traversals violates validate_shape."""
    client, _ = api
    resp = await client.post(
        "/v1/query-profiles",
        json={
            "profile_key": "bad_section",
            "label": "Bad Section",
            "kind": "section",
            "definition": {"traversals": []},
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_section_properties_empty_sections_422(api):
    """section_properties with empty profile_sections violates validate_shape."""
    client, _ = api
    resp = await client.post(
        "/v1/query-profiles",
        json={
            "profile_key": "bad_sp",
            "label": "Bad SP",
            "kind": "section_properties",
            "root_entity_types": ["RADAR_SYSTEM"],
            "definition": {"profile_sections": []},
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_section_properties_noncanonical_root_422(api):
    """section_properties root_entity_types must be RADAR/MISSILE only."""
    client, _ = api
    resp = await client.post(
        "/v1/query-profiles",
        json={
            "profile_key": "bad_root",
            "label": "Bad Root",
            "kind": "section_properties",
            "root_entity_types": ["FOO_SYSTEM"],
            "definition": {"profile_sections": ["rf_parameters"]},
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_dossier_without_refs_422(api):
    """kind=dossier with no section_profile_ids violates validate_shape."""
    client, _ = api
    resp = await client.post(
        "/v1/query-profiles",
        json={
            "profile_key": "bad_dossier",
            "label": "Bad Dossier",
            "kind": "dossier",
            "root_entity_types": ["RADAR_SYSTEM"],
            "definition": {"section_profile_ids": []},
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_dossier_nonexistent_ref_422(api):
    """A dossier referencing a profile_key that doesn't exist is rejected."""
    client, _ = api
    # get_profile → None for every lookup: the referenced section is missing.
    with patch(f"{_API}.get_profile", new=AsyncMock(return_value=None)):
        resp = await client.post(
            "/v1/query-profiles",
            json={
                "profile_key": "dossier_bad_ref",
                "label": "Dossier Bad Ref",
                "kind": "dossier",
                "root_entity_types": ["RADAR_SYSTEM"],
                "definition": {"section_profile_ids": ["ghost_section"]},
            },
        )
    assert resp.status_code == 422
    assert "ghost_section" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /v1/query-profiles/{profile_key}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_profile_found(api):
    client, _ = api
    row = _make_profile_row(profile_key="system_performance", label="System Performance")
    with patch(f"{_API}.get_required_profile", new=AsyncMock(return_value=row)):
        resp = await client.get("/v1/query-profiles/system_performance")
    assert resp.status_code == 200
    assert resp.json()["label"] == "System Performance"


@pytest.mark.asyncio
async def test_get_profile_not_found_404(api):
    from app.services.query_profiles import QueryProfileNotFoundError

    client, _ = api
    with patch(
        f"{_API}.get_required_profile",
        new=AsyncMock(side_effect=QueryProfileNotFoundError("nope")),
    ):
        resp = await client.get("/v1/query-profiles/missing")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /v1/query-profiles/{profile_key} -- partial update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_profile_partial(api):
    client, _ = api
    # existing row is a valid section_properties profile, so the merged shape
    # (existing + delta) stays valid.
    existing = _make_profile_row(profile_key="system_components")
    updated = _make_profile_row(
        profile_key="system_components", enabled=False, description="updated"
    )
    update_mock = AsyncMock(return_value=updated)
    with patch(
        f"{_API}.get_required_profile", new=AsyncMock(return_value=existing)
    ), patch(f"{_API}.update_profile", new=update_mock):
        resp = await client.put(
            "/v1/query-profiles/system_components",
            json={"enabled": False, "description": "updated"},
        )
    assert resp.status_code == 200
    # exclude_unset: only the two supplied fields are forwarded (source_id/kind
    # untouched so the service leaves them at their _UNSET default).
    kwargs = update_mock.await_args.kwargs
    assert set(kwargs) == {"enabled", "description"}
    assert kwargs["enabled"] is False


@pytest.mark.asyncio
async def test_update_profile_not_found_404(api):
    from app.services.query_profiles import QueryProfileNotFoundError

    client, _ = api
    with patch(
        f"{_API}.get_required_profile",
        new=AsyncMock(side_effect=QueryProfileNotFoundError("nope")),
    ):
        resp = await client.put(
            "/v1/query-profiles/missing", json={"label": "x"}
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_profile_makes_invalid_shape_422(api):
    """A partial update that would invalidate the MERGED profile is rejected:
    flip an existing section_properties profile to kind=section (which then has
    no traversals) → validate_shape fails."""
    client, _ = api
    existing = _make_profile_row(
        profile_key="system_components",
        kind="section_properties",
        definition={"profile_sections": ["components"]},
    )
    update_mock = AsyncMock()
    with patch(
        f"{_API}.get_required_profile", new=AsyncMock(return_value=existing)
    ), patch(f"{_API}.update_profile", new=update_mock):
        resp = await client.put(
            "/v1/query-profiles/system_components",
            json={"kind": "section"},
        )
    assert resp.status_code == 422
    # The service update was never reached — validation short-circuited.
    update_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# DELETE /v1/query-profiles/{profile_key}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_profile_204(api):
    client, _ = api
    del_mock = AsyncMock(return_value=None)
    with patch(f"{_API}.delete_profile", new=del_mock):
        resp = await client.delete("/v1/query-profiles/system_components")
    assert resp.status_code == 204
    assert del_mock.await_args.args[1] == "system_components"


@pytest.mark.asyncio
async def test_delete_profile_referenced_409(api):
    from app.services.query_profiles import QueryProfileReferencedError

    client, _ = api
    with patch(
        f"{_API}.delete_profile",
        new=AsyncMock(side_effect=QueryProfileReferencedError("still referenced")),
    ):
        resp = await client.delete("/v1/query-profiles/system_components")
    assert resp.status_code == 409
    assert "referenced" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_profile_not_found_404(api):
    from app.services.query_profiles import QueryProfileNotFoundError

    client, _ = api
    with patch(
        f"{_API}.delete_profile",
        new=AsyncMock(side_effect=QueryProfileNotFoundError("nope")),
    ):
        resp = await client.delete("/v1/query-profiles/missing")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/query-profiles/search/section + /dossier
# ---------------------------------------------------------------------------


def _section_response():
    from app.schemas.graph_store import GraphEntityResult
    from app.schemas.query_profiles import QueryProfileSectionResponse

    return QueryProfileSectionResponse(
        profile_id="system_components",
        profile_label="System Components",
        resolved_root=GraphEntityResult(name="SA-2", entity_type="MISSILE_SYSTEM"),
        items=[],
        total=0,
    )


def _dossier_response():
    from app.schemas.graph_store import GraphEntityResult
    from app.schemas.query_profiles import QueryProfileDossierResponse

    return QueryProfileDossierResponse(
        profile_id="system_dossier",
        profile_label="System Dossier",
        resolved_root=GraphEntityResult(name="SA-2", entity_type="MISSILE_SYSTEM"),
        sections=[],
    )


@pytest.mark.asyncio
async def test_search_section_happy_path(api):
    client, _ = api
    with patch(f"{_API}.get_graph_store", return_value=MagicMock()), patch(
        f"{_API}.execute_section_search",
        new=AsyncMock(return_value=_section_response()),
    ):
        resp = await client.post(
            "/v1/query-profiles/search/section",
            json={"profile_id": "system_components", "query_text": "SA-2"},
        )
    assert resp.status_code == 200
    assert resp.json()["profile_id"] == "system_components"


@pytest.mark.asyncio
async def test_search_section_profile_missing_404(api):
    from app.services.query_profiles import QueryProfileNotFoundError

    client, _ = api
    with patch(f"{_API}.get_graph_store", return_value=MagicMock()), patch(
        f"{_API}.execute_section_search",
        new=AsyncMock(side_effect=QueryProfileNotFoundError("nope")),
    ):
        resp = await client.post(
            "/v1/query-profiles/search/section",
            json={"profile_id": "missing", "query_text": "SA-2"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_dossier_happy_path(api):
    client, _ = api
    with patch(f"{_API}.get_graph_store", return_value=MagicMock()), patch(
        f"{_API}.execute_dossier_search",
        new=AsyncMock(return_value=_dossier_response()),
    ):
        resp = await client.post(
            "/v1/query-profiles/search/dossier",
            json={"profile_id": "system_dossier", "query_text": "SA-2"},
        )
    assert resp.status_code == 200
    assert resp.json()["profile_id"] == "system_dossier"


@pytest.mark.asyncio
async def test_search_dossier_root_not_found_404(api):
    from app.services.query_profiles import QueryRootNotFoundError

    client, _ = api
    with patch(f"{_API}.get_graph_store", return_value=MagicMock()), patch(
        f"{_API}.execute_dossier_search",
        new=AsyncMock(side_effect=QueryRootNotFoundError("no root")),
    ):
        resp = await client.post(
            "/v1/query-profiles/search/dossier",
            json={"profile_id": "system_dossier", "query_text": "SA-2"},
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Removed registry routes are gone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_list_route_gone(api):
    client, _ = api
    # GET /query-profiles/registries now falls through to GET {profile_key}
    # ("registries" as a key) which 404s — the registry list route is gone.
    with patch(
        f"{_API}.get_required_profile",
        new=AsyncMock(
            side_effect=__import__(
                "app.services.query_profiles", fromlist=["QueryProfileNotFoundError"]
            ).QueryProfileNotFoundError("nope")
        ),
    ):
        resp = await client.get("/v1/query-profiles/registries")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_registry_create_route_gone(api):
    client, _ = api
    # No POST handler exists for /query-profiles/{profile_key}, so POSTing the
    # old registry-create path yields 405 (method not allowed) — the route is
    # definitively gone.
    resp = await client.post("/v1/query-profiles/registries", json={})
    assert resp.status_code in (404, 405)


@pytest.mark.asyncio
async def test_default_template_route_gone(api):
    client, _ = api
    with patch(
        f"{_API}.get_required_profile",
        new=AsyncMock(
            side_effect=__import__(
                "app.services.query_profiles", fromlist=["QueryProfileNotFoundError"]
            ).QueryProfileNotFoundError("nope")
        ),
    ):
        resp = await client.get("/v1/query-profiles/default-template")
    assert resp.status_code == 404
