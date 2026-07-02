"""End-to-end source-scope gate for standalone query profiles.

Unlike ``test_query_profiles_api.py`` (which mocks every service + GraphStore),
this suite drives the **live** API on ``localhost:8005`` and the live ArcadeDB
graph behind it. It exists to PROVE that a profile's ``source_id`` scope genuinely
filters the exact-graph search, end to end.

Corpus facts (fixed for this environment):
  * The only ingest source is ``SA-2_Sources`` (id ``2578003b-…``); ALL ~21
    documents belong to it.
  * ``Fan Song`` resolves to a ``RADAR_SYSTEM`` root (canonical ``FAN SONG``)
    that lives in that corpus.

Because there is exactly ONE populated source, the source predicate is proven
with three arms:

  1. **Global** (``source_id`` null) — the root resolves and returns field-groups.
  2. **Scoped to the real source** — same root, and the returned field-group set
     is a SUBSET of Global (here equal, since every doc is in this source).
  3. **Scoped to a source with no documents** — the root has no in-source
     document, the predicate drops it, and the search 404s
     (``QueryRootNotFound``). This is the strict-subset proof (∅ ⊂ Global).

Note on arm 3: ``query_profiles.source_id`` carries a FK to ``ingest.sources.id``,
so a *random* UUID cannot be stored (create → 409, see
``test_random_uuid_source_rejected``). The faithful realization of "a source
with no documents" is therefore a real-but-empty source, created (and torn down)
directly in Postgres — the SEARCH assertions themselves stay 100% on the live API.

Skips cleanly when the live API is unreachable; the arm-3 / associated-systems
tests additionally skip if the production Postgres (for temp-source lifecycle)
is unreachable.
"""
from __future__ import annotations

import json
import socket
import time
import uuid

import pytest

pytestmark = pytest.mark.integration

API_HOST = "localhost"
API_PORT = 8005
BASE_URL = f"http://{API_HOST}:{API_PORT}"

REAL_SOURCE_ID = "2578003b-ce68-46d0-b52a-221902158bc6"  # SA-2_Sources
ROOT_QUERY = "Fan Song"
EXPECTED_CANONICAL = "FAN SONG"
EXPECTED_ROOT_TYPE = "RADAR_SYSTEM"
SYSTEM_USER = "00000000-0000-0000-0000-000000000001"

# The four profiles that must exist before and after this suite.
BASELINE_PROFILE_KEYS = {
    "system_rf_parameters",
    "system_components",
    "system_performance",
    "system_dossier",
}


# ---------------------------------------------------------------------------
# Reachability probes (mirrors tests/integration/conftest.py)
# ---------------------------------------------------------------------------


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pg_params() -> dict | None:
    """Connection params for the *production* Postgres the live API uses.

    Creds come from the app settings (which load ``.env``); the host is probed
    because in-container it is ``postgres`` but from the host it is ``localhost``.
    Returns ``None`` if unreachable — arm-3 tests skip in that case.
    """
    try:
        from app.config import get_settings

        settings = get_settings()
    except Exception:
        return None
    port = settings.postgres_port
    for host in ("localhost", "127.0.0.1", settings.postgres_host):
        if _reachable(host, port):
            return {
                "host": host,
                "port": port,
                "user": settings.postgres_user,
                "password": settings.postgres_password,
                "dbname": settings.postgres_db,
            }
    return None


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _content_cells(resp: dict) -> set[tuple]:
    """Deterministic field-group payload as a set of (subgroup, field, value)."""
    cells: set[tuple] = set()
    for group in resp.get("field_groups") or []:
        subgroup = group.get("subgroup")
        for field in group.get("fields", []):
            cells.add(
                (subgroup, field.get("name"), json.dumps(field.get("value"), sort_keys=True))
            )
    return cells


def _evidence_chunks(resp: dict) -> set[str]:
    chunks: set[str] = set()
    for group in resp.get("field_groups") or []:
        for field in group.get("fields", []):
            for ev in field.get("evidence", []):
                if ev.get("chunk_id"):
                    chunks.add(ev["chunk_id"])
    return chunks


def _related(resp: dict) -> set[tuple]:
    return {
        (x.get("canonical_name") or x.get("name"), x.get("entity_type"))
        for x in (resp.get("related_systems") or [])
    }


# ---------------------------------------------------------------------------
# Harness — live-API client + optional Postgres for temp-source lifecycle
# ---------------------------------------------------------------------------


class _Harness:
    def __init__(self, client, pg):
        self.client = client
        self.pg = pg  # psycopg2 connection (autocommit) or None
        self._profile_keys: list[str] = []
        self._source_ids: list[str] = []

    # --- profiles (live API) ---------------------------------------------
    def create_profile(self, **body) -> dict:
        resp = self.client.post("/v1/query-profiles", json=body)
        assert resp.status_code == 201, (
            f"create profile {body.get('profile_key')!r} -> "
            f"{resp.status_code}: {resp.text}"
        )
        key = body["profile_key"]
        self._profile_keys.append(key)
        # The live API commits in dependency teardown, so a just-created row is
        # not immediately visible to the next request (observed ~0.1-0.3s lag).
        # Wait for visibility so a later search 404 can only mean "root dropped",
        # never "profile not yet committed".
        assert self._wait_visible(key), f"profile {key!r} never became visible"
        return resp.json()

    def _wait_visible(self, key: str, tries: int = 60, delay: float = 0.1) -> bool:
        for _ in range(tries):
            if self.client.get(f"/v1/query-profiles/{key}").status_code == 200:
                return True
            time.sleep(delay)
        return False

    # --- temp source (direct Postgres; no delete endpoint exists) ---------
    def create_empty_source(self) -> str:
        assert self.pg is not None, "postgres required for empty-source lifecycle"
        sid = str(uuid.uuid4())
        cur = self.pg.cursor()
        cur.execute(
            "insert into ingest.sources (id, name, created_by) values (%s, %s, %s)",
            (sid, f"e2e_empty_{sid[:8]}", SYSTEM_USER),
        )
        self._source_ids.append(sid)
        return sid

    # --- searches (live API) ---------------------------------------------
    def search_section(self, profile_id: str, query: str = ROOT_QUERY):
        return self.client.post(
            "/v1/query-profiles/search/section",
            json={
                "profile_id": profile_id,
                "query_text": query,
                "include_evidence": True,
                "evidence_top_k": 3,
                "top_k": 25,
            },
        )

    def search_dossier(self, profile_id: str, query: str = ROOT_QUERY):
        return self.client.post(
            "/v1/query-profiles/search/dossier",
            json={"profile_id": profile_id, "query_text": query},
        )

    def cleanup(self) -> None:
        for key in self._profile_keys:
            try:
                self.client.delete(f"/v1/query-profiles/{key}")
            except Exception:
                pass
        if self.pg is not None:
            cur = self.pg.cursor()
            for sid in self._source_ids:
                try:
                    cur.execute("delete from ingest.sources where id = %s", (sid,))
                except Exception:
                    pass


@pytest.fixture
def harness():
    if not _reachable(API_HOST, API_PORT):
        pytest.skip(f"live api not reachable on {BASE_URL}")
    import httpx

    client = httpx.Client(base_url=BASE_URL, timeout=90.0)

    pg = None
    params = _pg_params()
    if params is not None:
        try:
            import psycopg2

            pg = psycopg2.connect(connect_timeout=3, **params)
            pg.autocommit = True
        except Exception:
            pg = None

    h = _Harness(client, pg)
    try:
        yield h
    finally:
        h.cleanup()
        client.close()
        if pg is not None:
            pg.close()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_baseline_four_profiles(harness):
    """The migrated corpus exposes exactly the four first-class profiles."""
    resp = harness.client.get("/v1/query-profiles")
    assert resp.status_code == 200
    keys = {p["profile_key"] for p in resp.json()}
    assert BASELINE_PROFILE_KEYS <= keys, f"missing baseline profiles: {keys}"


# ---------------------------------------------------------------------------
# Arm 1 — Global
# ---------------------------------------------------------------------------


def test_global_section_resolves_root(harness):
    """Global (source_id null) resolves the Fan Song root + returns field-groups."""
    resp = harness.search_section("system_rf_parameters")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    root = data["resolved_root"]
    assert root["canonical_name"] == EXPECTED_CANONICAL
    assert root["entity_type"] == EXPECTED_ROOT_TYPE
    assert data["field_groups"], "global search returned no field-groups"
    assert _content_cells(data), "global search returned no field cells"


# ---------------------------------------------------------------------------
# Arm 2 — Scoped to the real source (subset, non-empty)
# ---------------------------------------------------------------------------


def test_real_source_scope_is_nonempty_subset_of_global(harness):
    """A profile scoped to the real source returns the SAME root and a
    field-group set ⊆ Global (equal here, since every doc is in this source)."""
    global_resp = harness.search_section("system_rf_parameters")
    assert global_resp.status_code == 200, global_resp.text
    global_data = global_resp.json()
    global_cells = _content_cells(global_data)
    global_chunks = _evidence_chunks(global_data)

    key = f"tmp_e2e_rf_real_{uuid.uuid4().hex[:8]}"
    harness.create_profile(
        profile_key=key,
        label="e2e real-source rf_parameters",
        kind="section_properties",
        root_entity_types=[EXPECTED_ROOT_TYPE, "MISSILE_SYSTEM"],
        definition={"profile_sections": ["rf_parameters"]},
        source_id=REAL_SOURCE_ID,
        enabled=True,
    )

    scoped_resp = harness.search_section(key)
    assert scoped_resp.status_code == 200, scoped_resp.text
    scoped_data = scoped_resp.json()

    # Same resolved root as Global.
    assert scoped_data["resolved_root"]["canonical_name"] == EXPECTED_CANONICAL
    assert scoped_data["resolved_root"]["node_id"] == global_data["resolved_root"]["node_id"]

    scoped_cells = _content_cells(scoped_data)
    # Non-empty AND a subset of Global (no false exclusion — all docs in-source).
    assert scoped_cells, "real-source scope returned empty field-groups"
    assert scoped_cells <= global_cells, (
        f"scoped cells not ⊆ global: extra={scoped_cells - global_cells}"
    )
    assert _evidence_chunks(scoped_data) <= global_chunks


# ---------------------------------------------------------------------------
# Arm 3 — Scoped to a source with no documents (strict subset ∅ ⊂ Global)
# ---------------------------------------------------------------------------


def test_empty_source_scope_drops_root_404(harness):
    """Scoping to a real-but-empty source drops the root: no in-source document
    for Fan Song, so the source predicate excludes it and the search 404s.

    This is the strict-subset proof: Global resolves (∈), empty-source ∉ (∅)."""
    if harness.pg is None:
        pytest.skip("production postgres unreachable (needed for temp-source lifecycle)")

    # Sanity: Global still resolves the root that the empty scope must drop.
    assert harness.search_section("system_rf_parameters").status_code == 200

    empty_source = harness.create_empty_source()
    key = f"tmp_e2e_rf_empty_{uuid.uuid4().hex[:8]}"
    harness.create_profile(
        profile_key=key,
        label="e2e empty-source rf_parameters",
        kind="section_properties",
        root_entity_types=[EXPECTED_ROOT_TYPE, "MISSILE_SYSTEM"],
        definition={"profile_sections": ["rf_parameters"]},
        source_id=empty_source,
        enabled=True,
    )

    resp = harness.search_section(key)
    assert resp.status_code == 404, (
        f"expected 404 (root dropped by source filter), got {resp.status_code}: {resp.text}"
    )
    # The 404 must be QueryRootNotFound (predicate dropped the root), NOT a
    # missing-profile 404 (the create+wait_visible above guarantees the profile
    # exists). QueryRootNotFoundError renders as "No matching root entity found".
    detail = resp.json().get("detail", "").lower()
    assert "root" in detail, f"404 not from root-drop predicate: {detail!r}"
    assert "does not exist" not in detail, f"404 was profile-missing, not root-drop: {detail!r}"


def test_random_uuid_source_rejected(harness):
    """A random (non-existent) source UUID cannot be stored — the FK to
    ingest.sources rejects it (409). Documents why arm 3 uses a real empty
    source rather than a random UUID."""
    resp = harness.client.post(
        "/v1/query-profiles",
        json={
            "profile_key": f"tmp_e2e_rand_{uuid.uuid4().hex[:8]}",
            "label": "e2e random source",
            "kind": "section_properties",
            "root_entity_types": [EXPECTED_ROOT_TYPE],
            "definition": {"profile_sections": ["rf_parameters"]},
            "source_id": str(uuid.uuid4()),
        },
    )
    # Defensive: if the FK were ever dropped and this somehow persisted, track
    # it for teardown so the DB still ends clean.
    if resp.status_code == 201:
        harness._profile_keys.append(resp.json()["profile_key"])
    assert resp.status_code == 409, f"expected FK-violation 409, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Associated systems — real-source = full, empty-source = none
# ---------------------------------------------------------------------------


def test_associated_systems_respect_source_scope(harness):
    """system_components (include_associated_systems=true): the associated-system
    set is full under the real source and dropped entirely under an empty one."""
    if harness.pg is None:
        pytest.skip("production postgres unreachable (needed for temp-source lifecycle)")

    global_resp = harness.search_section("system_components")
    assert global_resp.status_code == 200, global_resp.text
    global_related = _related(global_resp.json())
    assert global_related, "system_components global search returned no associated systems"

    # Real-source scope: same associated systems as Global.
    real_key = f"tmp_e2e_comp_real_{uuid.uuid4().hex[:8]}"
    harness.create_profile(
        profile_key=real_key,
        label="e2e real-source components",
        kind="section_properties",
        root_entity_types=[EXPECTED_ROOT_TYPE, "MISSILE_SYSTEM"],
        definition={"profile_sections": ["components"], "include_associated_systems": True},
        source_id=REAL_SOURCE_ID,
        enabled=True,
    )
    real_resp = harness.search_section(real_key)
    assert real_resp.status_code == 200, real_resp.text
    real_related = _related(real_resp.json())
    assert real_related, "real-source components returned no associated systems"
    assert real_related <= global_related

    # Empty-source scope: root dropped, so no associated systems at all (404).
    empty_source = harness.create_empty_source()
    empty_key = f"tmp_e2e_comp_empty_{uuid.uuid4().hex[:8]}"
    harness.create_profile(
        profile_key=empty_key,
        label="e2e empty-source components",
        kind="section_properties",
        root_entity_types=[EXPECTED_ROOT_TYPE, "MISSILE_SYSTEM"],
        definition={"profile_sections": ["components"], "include_associated_systems": True},
        source_id=empty_source,
        enabled=True,
    )
    empty_resp = harness.search_section(empty_key)
    assert empty_resp.status_code == 404, (
        f"empty-source components expected 404, got {empty_resp.status_code}: {empty_resp.text}"
    )


# ---------------------------------------------------------------------------
# Smoke — every migrated profile answers without a 500
# ---------------------------------------------------------------------------


def test_all_migrated_profiles_smoke_no_500(harness):
    """All four migrated profiles run their search endpoint with no 500s."""
    section_profiles = ["system_rf_parameters", "system_components", "system_performance"]
    for profile_id in section_profiles:
        resp = harness.search_section(profile_id)
        assert resp.status_code < 500, f"{profile_id} section -> {resp.status_code}: {resp.text}"
        assert resp.status_code == 200, f"{profile_id} section -> {resp.status_code}: {resp.text}"

    dossier_resp = harness.search_dossier("system_dossier")
    assert dossier_resp.status_code < 500, dossier_resp.text
    assert dossier_resp.status_code == 200, dossier_resp.text
