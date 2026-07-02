"""Migration 0022 — governance.query_profiles table + merge-preserving
data migration off the query_profile_registries row.

Exercises the extracted ``_seed_query_profiles`` function against a real
Postgres connection inside the ``db_session`` rollback transaction, so the
merge behavior is verified without mutating persisted data.

Two paths:
  (a) empty/absent registry profiles -> the 4 canonical starters are seeded
      with real labels, and system_dossier.definition->section_profile_ids
      references the other three profile_keys.
  (b) a registry row carrying a custom profile (id not in canonical) ->
      the custom profile is PRESERVED and the missing canonical starters
      are added (merge, not replace).

If the db_session fixture can't reach test-postgres (known repo infra issue:
:5433 occupied by an unrelated service), run pointed at the live DB via
DATABASE_URL_SYNC — the fixture's rollback transaction protects it.
"""
import importlib.util
import json
from pathlib import Path

from sqlalchemy import text

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "0022_query_profiles_table.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_mig_0022", _MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MIG = _load_migration()


def _ensure_registry_table(conn):
    """Create a minimal ``governance.query_profile_registries`` table if it
    is absent.

    Migration 0023 drops the real registry table, so once the DB is at head
    this table no longer exists — but 0022's ``_seed_query_profiles`` still
    reads it (``SELECT profiles ... WHERE is_active ... ORDER BY updated_at``).
    This recreates just the columns that seed function (and these tests)
    touch, so the migration's frozen seed logic can be exercised regardless
    of whether 0023 has already dropped the real table. Inside the
    ``db_session`` rollback transaction this DDL is transactional and rolls
    back with the fixture.
    """
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS governance.query_profile_registries ("
            "  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
            "  profiles jsonb NOT NULL DEFAULT '[]'::jsonb,"
            "  is_active boolean NOT NULL DEFAULT false,"
            "  updated_at timestamptz NOT NULL DEFAULT now()"
            ")"
        )
    )


def _reset(conn):
    """Clear the target table + neutralize existing registry rows so the
    seed function starts from a known-empty registry state."""
    _ensure_registry_table(conn)
    conn.execute(text("DELETE FROM governance.query_profiles"))
    conn.execute(
        text(
            "UPDATE governance.query_profile_registries "
            "SET is_active = false, profiles = '[]'::jsonb"
        )
    )


def test_seed_empty_registry_creates_canonical_starters(db_session):
    """Path (a): no registry profiles -> 4 canonical starters seeded."""
    conn = db_session.connection()
    _reset(conn)

    _MIG._seed_query_profiles(conn)

    rows = conn.execute(
        text(
            "SELECT profile_key, label, kind, enabled, source_id "
            "FROM governance.query_profiles ORDER BY profile_key"
        )
    ).fetchall()
    by_key = {r[0]: r for r in rows}

    for key in (
        "system_components",
        "system_dossier",
        "system_performance",
        "system_rf_parameters",
    ):
        assert key in by_key, f"canonical starter {key} missing"

    # Real labels carried over, not blanks.
    assert by_key["system_components"][1] == "System Components"
    assert by_key["system_dossier"][2] == "dossier"
    # enabled default TRUE, source_id NULL.
    assert all(r[3] is True for r in rows)
    assert all(r[4] is None for r in rows)

    # Dossier's definition references the other three profile_keys.
    dossier_def = conn.execute(
        text(
            "SELECT definition FROM governance.query_profiles "
            "WHERE profile_key = 'system_dossier'"
        )
    ).scalar_one()
    if isinstance(dossier_def, str):
        dossier_def = json.loads(dossier_def)
    assert set(dossier_def["section_profile_ids"]) == {
        "system_rf_parameters",
        "system_components",
        "system_performance",
    }

    # placeholder_query round-trips into definition (item 1).
    comp_def = conn.execute(
        text(
            "SELECT definition->>'placeholder_query' "
            "FROM governance.query_profiles WHERE profile_key = 'system_components'"
        )
    ).scalar_one()
    assert comp_def == "e.g. SA-2"


def test_seed_preserves_custom_profile_and_adds_missing(db_session):
    """Path (b): a custom registry profile is preserved AND the missing
    canonical starters are added (merge-preserving behavior)."""
    conn = db_session.connection()
    _reset(conn)

    custom = {
        "id": "custom_test_profile_0022",
        "label": "Custom Test Profile",
        "description": "A user-authored profile that must not be discarded.",
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM"],
        "profile_sections": ["custom_section"],
        "placeholder_query": "e.g. custom",
    }
    conn.execute(
        text(
            "INSERT INTO governance.query_profile_registries "
            "(id, profiles, is_active, updated_at) "
            "VALUES (gen_random_uuid(), CAST(:profiles AS jsonb), true, now())"
        ),
        {"profiles": json.dumps([custom])},
    )

    _MIG._seed_query_profiles(conn)

    keys = {
        r[0]
        for r in conn.execute(
            text("SELECT profile_key FROM governance.query_profiles")
        ).fetchall()
    }
    # Custom preserved.
    assert "custom_test_profile_0022" in keys
    # All 4 canonical starters added.
    assert {
        "system_components",
        "system_dossier",
        "system_performance",
        "system_rf_parameters",
    } <= keys
    # 1 custom + 4 canonical.
    assert len(keys) == 5

    # Custom profile's placeholder_query preserved into definition.
    custom_ph = conn.execute(
        text(
            "SELECT definition->>'placeholder_query' "
            "FROM governance.query_profiles "
            "WHERE profile_key = 'custom_test_profile_0022'"
        )
    ).scalar_one()
    assert custom_ph == "e.g. custom"
