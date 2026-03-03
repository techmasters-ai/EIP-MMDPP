#!/usr/bin/env python3
"""Idempotent ontology seeder.

Loads entity types and relationship types from ontology/base_v1.yaml
into the ontology schema. Safe to run multiple times (upsert by name).
"""

import os
import sys
import uuid
from pathlib import Path

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.base import Base
from app.models.governance import Feedback, Patch  # noqa (register models)
from app.models.ingest import Source  # noqa
from app.models.retrieval import Chunk  # noqa

settings = get_settings()

ONTOLOGY_FILE = Path(__file__).parent.parent / "ontology" / "base_v1.yaml"


def seed(session: Session) -> None:
    # Dynamic import to avoid circular issues at module level
    from sqlalchemy import Column, String, Boolean, DateTime, Text
    from sqlalchemy.dialects.postgresql import UUID, JSONB

    # We use raw SQL inserts for the ontology tables since they're in
    # the 'ontology' schema and not registered as ORM models in this script.

    with open(ONTOLOGY_FILE) as f:
        data = yaml.safe_load(f)

    version_number = data["version"]

    # Upsert version record
    existing_version = session.execute(
        text("SELECT id FROM ontology.versions WHERE version_number = :v"),
        {"v": version_number},
    ).fetchone()

    if existing_version:
        version_id = existing_version[0]
        print(f"Ontology version {version_number} already exists (id={version_id})")
    else:
        version_id = str(uuid.uuid4())
        session.execute(
            text("""
                INSERT INTO ontology.versions (id, version_number, is_current, notes)
                VALUES (:id, :version, true, 'Base military equipment ontology')
                ON CONFLICT (version_number) DO NOTHING
            """),
            {"id": version_id, "version": version_number},
        )
        # Mark all other versions as not current
        session.execute(
            text("""
                UPDATE ontology.versions
                SET is_current = false
                WHERE version_number != :v
            """),
            {"v": version_number},
        )
        print(f"Created ontology version {version_number} (id={version_id})")

    # Upsert entity types
    entity_type_ids: dict[str, str] = {}
    for et in data.get("entity_types", []):
        existing = session.execute(
            text("SELECT id FROM ontology.entity_types WHERE name = :name"),
            {"name": et["name"]},
        ).fetchone()

        if existing:
            entity_type_ids[et["name"]] = str(existing[0])
            print(f"  Entity type exists: {et['name']}")
        else:
            et_id = str(uuid.uuid4())
            import json
            session.execute(
                text("""
                    INSERT INTO ontology.entity_types
                        (id, name, label, description, properties, version_id, is_abstract)
                    VALUES
                        (:id, :name, :label, :description, :properties::jsonb, :version_id, :is_abstract)
                    ON CONFLICT (name) DO UPDATE SET
                        label = EXCLUDED.label,
                        description = EXCLUDED.description,
                        properties = EXCLUDED.properties
                """),
                {
                    "id": et_id,
                    "name": et["name"],
                    "label": et["label"],
                    "description": et.get("description", ""),
                    "properties": json.dumps(et.get("properties", {})),
                    "version_id": version_id,
                    "is_abstract": et.get("is_abstract", False),
                },
            )
            entity_type_ids[et["name"]] = et_id
            print(f"  Created entity type: {et['name']}")

    # Upsert relationship types
    for rt in data.get("relationship_types", []):
        existing = session.execute(
            text("SELECT id FROM ontology.relationship_types WHERE name = :name"),
            {"name": rt["name"]},
        ).fetchone()

        source_type_id = entity_type_ids.get(rt.get("source_type")) if rt.get("source_type") else None
        target_type_id = entity_type_ids.get(rt.get("target_type")) if rt.get("target_type") else None

        if existing:
            print(f"  Relationship type exists: {rt['name']}")
        else:
            rt_id = str(uuid.uuid4())
            session.execute(
                text("""
                    INSERT INTO ontology.relationship_types
                        (id, name, label, description, source_type_id, target_type_id, cardinality, version_id)
                    VALUES
                        (:id, :name, :label, :description, :source_type_id, :target_type_id, :cardinality, :version_id)
                    ON CONFLICT (name) DO UPDATE SET
                        label = EXCLUDED.label,
                        description = EXCLUDED.description
                """),
                {
                    "id": rt_id,
                    "name": rt["name"],
                    "label": rt["label"],
                    "description": rt.get("description", ""),
                    "source_type_id": source_type_id,
                    "target_type_id": target_type_id,
                    "cardinality": rt.get("cardinality", "many_to_many"),
                    "version_id": version_id,
                },
            )
            print(f"  Created relationship type: {rt['name']}")

    session.commit()
    print(f"\nOntology seeding complete. Version: {version_number}")


def main() -> None:
    engine = create_engine(settings.sync_database_url)
    with Session(engine) as session:
        try:
            seed(session)
        except Exception as e:
            session.rollback()
            print(f"Ontology seeding failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
