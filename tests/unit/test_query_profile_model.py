"""Model-level tests for QueryProfile (Task 0 of the standalone-query-profiles
plan). No live DB — inspects the declared SQLAlchemy columns directly.

Run standalone:
    python3 -m pytest tests/unit/test_query_profile_model.py -v
"""
from app.models.query_profiles import QueryProfile


def _column(name):
    return QueryProfile.__table__.columns[name]


def test_tablename_and_schema():
    assert QueryProfile.__tablename__ == "query_profiles"
    assert QueryProfile.__table__.schema == "governance"
    assert QueryProfile.__table__.fullname == "governance.query_profiles"


def test_profile_key_unique_and_not_null():
    col = _column("profile_key")
    assert col.nullable is False
    assert col.unique is True or any(
        set(c.columns) == {col}
        for c in QueryProfile.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    )


def test_label_not_null():
    col = _column("label")
    assert col.nullable is False


def test_kind_not_null():
    col = _column("kind")
    assert col.nullable is False


def test_source_id_nullable():
    col = _column("source_id")
    assert col.nullable is True


def test_source_id_foreign_key_to_sources():
    col = _column("source_id")
    assert len(col.foreign_keys) == 1
    fk = next(iter(col.foreign_keys))
    assert fk.target_fullname == "ingest.sources.id"


def test_enabled_not_null_default_true():
    col = _column("enabled")
    assert col.nullable is False
    # Python-side default must be present and True (not merely absent).
    assert col.default is not None and col.default.arg is True


def test_root_entity_types_and_definition_are_jsonb():
    from sqlalchemy.dialects.postgresql import JSONB

    assert isinstance(_column("root_entity_types").type, JSONB)
    assert isinstance(_column("definition").type, JSONB)


def test_id_is_primary_key():
    col = _column("id")
    assert col.primary_key is True


def test_created_by_nullable():
    col = _column("created_by")
    assert col.nullable is True


def test_timestamps_present_and_not_null():
    for name in ("created_at", "updated_at"):
        col = _column(name)
        assert col.nullable is False
