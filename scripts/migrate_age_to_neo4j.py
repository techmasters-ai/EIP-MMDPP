#!/usr/bin/env python3
"""Migrate Apache AGE graph data to Neo4j.

Reads all nodes and edges from the AGE 'eip_kg' graph in PostgreSQL
and writes them to Neo4j using MERGE (idempotent).

Usage:
    python scripts/migrate_age_to_neo4j.py

Requires both PostgreSQL (with AGE) and Neo4j to be running.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def main():
    settings = get_settings()

    # Connect to PostgreSQL
    from sqlalchemy import create_engine, text

    engine = create_engine(settings.sync_database_url)

    # Connect to Neo4j
    from neo4j import GraphDatabase

    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    print("Reading nodes from AGE...")
    nodes = []
    edges = []

    try:
        with engine.connect() as conn:
            conn.execute(text("LOAD 'age'"))
            conn.execute(text("SET search_path = ag_catalog, \"$user\", public"))

            # Read all nodes
            result = conn.execute(text(
                "SELECT * FROM cypher('eip_kg', $$MATCH (n) RETURN n$$) AS (node agtype)"
            ))
            for row in result.fetchall():
                try:
                    node_data = json.loads(str(row[0]))
                    if isinstance(node_data, dict):
                        nodes.append(node_data)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Read all edges
            result = conn.execute(text(
                "SELECT * FROM cypher('eip_kg', "
                "$$MATCH (a)-[r]->(b) RETURN a.name, label(a), type(r), b.name, label(b), "
                "r.confidence, r.artifact_id$$) "
                "AS (from_name agtype, from_type agtype, rel_type agtype, "
                "to_name agtype, to_type agtype, confidence agtype, artifact_id agtype)"
            ))
            for row in result.fetchall():
                try:
                    edges.append({
                        "from_name": str(row[0]).strip('"'),
                        "from_type": str(row[1]).strip('"'),
                        "rel_type": str(row[2]).strip('"'),
                        "to_name": str(row[3]).strip('"'),
                        "to_type": str(row[4]).strip('"'),
                        "confidence": float(str(row[5])) if row[5] else 0.5,
                        "artifact_id": str(row[6]).strip('"') if row[6] else "",
                    })
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"WARNING: Could not read AGE graph (may not exist): {e}")
        print("No data to migrate.")
        neo4j_driver.close()
        return

    print(f"Found {len(nodes)} nodes and {len(edges)} edges in AGE")

    if not nodes and not edges:
        print("Nothing to migrate.")
        neo4j_driver.close()
        return

    # Write to Neo4j
    print("Writing nodes to Neo4j...")
    node_count = 0
    with neo4j_driver.session() as session:
        for node in nodes:
            name = node.get("name") or node.get("title") or str(node.get("id", ""))
            entity_type = node.get("entity_type") or node.get("label") or "Entity"

            if not name:
                continue

            # Clean properties for Neo4j (flatten nested dicts)
            props = {k: v for k, v in node.items() if isinstance(v, (str, int, float, bool))}
            props["name"] = name
            props["entity_type"] = entity_type

            try:
                session.run(
                    f"MERGE (n:Entity:{entity_type} {{name: $name, entity_type: $entity_type}}) "
                    "ON CREATE SET n += $props",
                    name=name,
                    entity_type=entity_type,
                    props=props,
                )
                node_count += 1
            except Exception as e:
                print(f"  Failed to migrate node {name}: {e}")

    print(f"Migrated {node_count} nodes")

    print("Writing edges to Neo4j...")
    edge_count = 0
    with neo4j_driver.session() as session:
        for edge in edges:
            try:
                session.run(
                    f"MATCH (a:Entity {{name: $from_name}}) "
                    f"MATCH (b:Entity {{name: $to_name}}) "
                    f"MERGE (a)-[r:{edge['rel_type']}]->(b) "
                    "SET r.confidence = $confidence, r.artifact_id = $artifact_id",
                    from_name=edge["from_name"],
                    to_name=edge["to_name"],
                    confidence=edge["confidence"],
                    artifact_id=edge["artifact_id"],
                )
                edge_count += 1
            except Exception as e:
                print(f"  Failed to migrate edge {edge['from_name']}→{edge['to_name']}: {e}")

    print(f"Migrated {edge_count} edges")
    neo4j_driver.close()
    print("AGE → Neo4j migration complete.")


if __name__ == "__main__":
    main()
