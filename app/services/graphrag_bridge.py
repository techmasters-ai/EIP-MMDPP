"""Neo4j/Postgres -> GraphRAG Parquet bridge layer.

Exports entities, relationships, text units, and documents from the
ontology-populated Neo4j graph and Postgres chunk store into pandas
DataFrames matching Microsoft GraphRAG's expected Parquet schema.
"""

import logging
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.services.ontology_templates import load_ontology

logger = logging.getLogger(__name__)

# Load scoring weights from ontology
_ontology = load_ontology()
_scoring_weights: dict[str, float] = _ontology.get("scoring_weights", {})
_DEFAULT_WEIGHT = _scoring_weights.get("default", 0.70)

# GraphRAG expected column schemas
_ENTITY_COLUMNS = ["id", "title", "type", "description", "human_readable_id"]
_RELATIONSHIP_COLUMNS = [
    "id", "source", "target", "description", "weight", "human_readable_id",
]
_TEXT_UNIT_COLUMNS = [
    "id", "text", "n_tokens", "document_ids", "entity_ids", "relationship_ids",
]
_DOCUMENT_COLUMNS = ["id", "title", "raw_content", "text_unit_ids"]


def export_entities(neo4j_driver) -> pd.DataFrame:
    """Export all Entity nodes from Neo4j into a GraphRAG entities DataFrame."""
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) "
                "RETURN n.name AS name, n.entity_type AS entity_type, "
                "n.description AS description, n.id AS id"
            )
            rows = []
            for i, record in enumerate(result):
                data = record.data()
                rows.append({
                    "id": data.get("id") or str(uuid.uuid4()),
                    "title": data.get("name", ""),
                    "type": data.get("entity_type", "UNKNOWN"),
                    "description": data.get("description", ""),
                    "human_readable_id": i,
                })
    except Exception:
        logger.exception("Failed to export entities from Neo4j")
        rows = []

    return pd.DataFrame(rows, columns=_ENTITY_COLUMNS)


def export_relationships(neo4j_driver) -> pd.DataFrame:
    """Export all relationships from Neo4j into a GraphRAG relationships DataFrame."""
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.name AS source, b.name AS target, "
                "type(r) AS relationship, r.description AS description"
            )
            rows = []
            for i, record in enumerate(result):
                data = record.data()
                rel_type = data.get("relationship", "")
                weight = _scoring_weights.get(rel_type, _DEFAULT_WEIGHT)
                rows.append({
                    "id": str(uuid.uuid4()),
                    "source": data.get("source", ""),
                    "target": data.get("target", ""),
                    "description": data.get("description") or rel_type,
                    "weight": weight,
                    "human_readable_id": i,
                })
    except Exception:
        logger.exception("Failed to export relationships from Neo4j")
        rows = []

    return pd.DataFrame(rows, columns=_RELATIONSHIP_COLUMNS)


def export_text_units(db_session) -> pd.DataFrame:
    """Export text chunks from Postgres retrieval.text_chunks."""
    try:
        result = db_session.execute(text(
            "SELECT id::text, chunk_text, document_id::text "
            "FROM retrieval.text_chunks "
            "WHERE chunk_text IS NOT NULL"
        ))
        rows = []
        for row in result.fetchall():
            chunk_id, content, doc_id = row
            rows.append({
                "id": chunk_id,
                "text": content or "",
                "n_tokens": len((content or "").split()),
                "document_ids": [doc_id] if doc_id else [],
                "entity_ids": [],
                "relationship_ids": [],
            })
    except Exception:
        logger.exception("Failed to export text units from Postgres")
        rows = []

    return pd.DataFrame(rows, columns=_TEXT_UNIT_COLUMNS)


def export_documents(db_session) -> pd.DataFrame:
    """Export document metadata from Postgres ingest.documents."""
    try:
        result = db_session.execute(text(
            "SELECT id::text, COALESCE(filename, 'untitled') "
            "FROM ingest.documents"
        ))
        rows = []
        for row in result.fetchall():
            doc_id, title = row
            rows.append({
                "id": doc_id,
                "title": title,
                "raw_content": "",
                "text_unit_ids": [],
            })
    except Exception:
        logger.exception("Failed to export documents from Postgres")
        rows = []

    return pd.DataFrame(rows, columns=_DOCUMENT_COLUMNS)


def export_all(neo4j_driver, db_session, output_dir: Path) -> dict:
    """Export all data and write Parquet files to output_dir.

    Returns dict with counts of exported entities, relationships, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entities_df = export_entities(neo4j_driver)
    relationships_df = export_relationships(neo4j_driver)
    text_units_df = export_text_units(db_session)
    documents_df = export_documents(db_session)

    # Link text_unit_ids back to documents
    if not text_units_df.empty and not documents_df.empty:
        doc_chunks = text_units_df.explode("document_ids")
        for doc_id in documents_df["id"]:
            chunk_ids = doc_chunks[
                doc_chunks["document_ids"] == doc_id
            ]["id"].tolist()
            documents_df.loc[
                documents_df["id"] == doc_id, "text_unit_ids"
            ] = [chunk_ids]

    entities_df.to_parquet(output_dir / "entities.parquet", index=False)
    relationships_df.to_parquet(output_dir / "relationships.parquet", index=False)
    text_units_df.to_parquet(output_dir / "text_units.parquet", index=False)
    documents_df.to_parquet(output_dir / "documents.parquet", index=False)

    stats = {
        "entities": len(entities_df),
        "relationships": len(relationships_df),
        "text_units": len(text_units_df),
        "documents": len(documents_df),
    }
    logger.info("GraphRAG bridge export complete: %s", stats)
    return stats
