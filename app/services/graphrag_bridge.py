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
_ENTITY_COLUMNS = ["id", "title", "type", "description", "human_readable_id", "text_unit_ids", "degree"]
_RELATIONSHIP_COLUMNS = [
    "id", "source", "target", "description", "weight", "human_readable_id", "text_unit_ids",
]
_TEXT_UNIT_COLUMNS = [
    "id", "text", "n_tokens", "document_id", "document_ids", "entity_ids", "relationship_ids",
]
_DOCUMENT_COLUMNS = ["id", "title", "raw_content", "text_unit_ids"]


def _synthesize_description(props: dict) -> str:
    """Build a readable description from entity properties."""
    name = props.get("name", "")
    etype = props.get("entity_type", "")
    desc = props.get("description", "")
    if desc:
        return desc

    # Build from available properties
    parts = [f"{name} is a {etype}."]
    skip = {"name", "entity_type", "id", "artifact_id", "last_artifact_id", "confidence", "description"}
    for k, v in props.items():
        if k in skip or v is None or v == "":
            continue
        label = k.replace("_", " ").title()
        parts.append(f"{label}: {v}")
    return " ".join(parts)


def export_entities(neo4j_driver) -> pd.DataFrame:
    """Export ALL nodes from Neo4j into a GraphRAG entities DataFrame.

    Includes Entity nodes, Document nodes, ChunkRef nodes, Alias nodes —
    everything in the graph. Synthesizes descriptions from node properties
    and populates text_unit_ids from EXTRACTED_FROM -> ChunkRef links.
    """
    try:
        with neo4j_driver.session() as session:
            # Get ALL nodes (not just :Entity)
            result = session.run(
                "MATCH (n) "
                "WHERE NOT n:ChunkRef "  # Exclude ChunkRef — they become text_unit_ids
                "RETURN n, labels(n) AS labels"
            )
            rows = []
            node_id_map: dict[str, str] = {}  # elementId -> stable id

            for i, record in enumerate(result):
                node = record["n"]
                props = dict(node)
                labels = record["labels"]

                # Use node's id property, or generate from elementId
                nid = props.get("id") or str(uuid.uuid5(uuid.NAMESPACE_URL, str(node.element_id)))
                name = props.get("name") or props.get("filename") or props.get("title") or str(labels[0])
                etype = props.get("entity_type") or labels[0]

                node_id_map[str(node.element_id)] = nid

                rows.append({
                    "id": nid,
                    "title": name,
                    "type": etype,
                    "description": _synthesize_description(props),
                    "human_readable_id": i,
                    "text_unit_ids": [],
                    "degree": 0,
                })

            # Populate text_unit_ids from EXTRACTED_FROM -> ChunkRef
            result = session.run(
                "MATCH (n)-[:EXTRACTED_FROM]->(c:ChunkRef) "
                "WHERE NOT n:ChunkRef "
                "RETURN n.id AS entity_id, c.chunk_id AS chunk_id"
            )
            entity_chunks: dict[str, list[str]] = {}
            for record in result:
                eid = record["entity_id"]
                cid = record["chunk_id"]
                if eid and cid:
                    entity_chunks.setdefault(eid, []).append(str(cid))

            # Compute degree (all relationships, not just Entity-Entity)
            result = session.run(
                "MATCH (n)-[r]-(m) "
                "WHERE NOT n:ChunkRef AND NOT m:ChunkRef "
                "RETURN n.id AS entity_id, count(r) AS degree"
            )
            entity_degrees: dict[str, int] = {}
            for record in result:
                if record["entity_id"]:
                    entity_degrees[record["entity_id"]] = record["degree"]

            for row in rows:
                row["text_unit_ids"] = entity_chunks.get(row["id"], [])
                row["degree"] = entity_degrees.get(row["id"], 0)

    except Exception:
        logger.exception("Failed to export entities from Neo4j")
        rows = []

    return pd.DataFrame(rows, columns=_ENTITY_COLUMNS)


def export_relationships(neo4j_driver) -> pd.DataFrame:
    """Export ALL relationships from Neo4j into a GraphRAG relationships DataFrame.

    Includes all edges except EXTRACTED_FROM (those become text_unit_ids instead).
    """
    try:
        with neo4j_driver.session() as session:
            # Get chunk mappings for text_unit_ids
            chunk_result = session.run(
                "MATCH (n)-[:EXTRACTED_FROM]->(c:ChunkRef) "
                "WHERE NOT n:ChunkRef "
                "RETURN n.name AS name, c.chunk_id AS chunk_id"
            )
            node_chunks: dict[str, list[str]] = {}
            for record in chunk_result:
                name = record["name"]
                cid = record["chunk_id"]
                if name and cid:
                    node_chunks.setdefault(name, []).append(str(cid))

            # Export ALL relationships except EXTRACTED_FROM and between ChunkRefs
            result = session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE NOT a:ChunkRef AND NOT b:ChunkRef "
                "AND type(r) <> 'EXTRACTED_FROM' "
                "RETURN coalesce(a.name, a.filename, labels(a)[0]) AS source, "
                "coalesce(b.name, b.filename, labels(b)[0]) AS target, "
                "type(r) AS relationship, r.description AS description"
            )
            rows = []
            for i, record in enumerate(result):
                data = record.data()
                src = data.get("source", "")
                tgt = data.get("target", "")
                rel_type = data.get("relationship", "")
                weight = _scoring_weights.get(rel_type, _DEFAULT_WEIGHT)

                shared_chunks = list(set(
                    node_chunks.get(src, []) + node_chunks.get(tgt, [])
                ))

                rows.append({
                    "id": str(uuid.uuid4()),
                    "source": src,
                    "target": tgt,
                    "description": data.get("description") or f"{src} {rel_type} {tgt}",
                    "weight": weight,
                    "human_readable_id": i,
                    "text_unit_ids": shared_chunks,
                })

            # Add co-occurrence edges: entities that share the same ChunkRef
            # are likely related. This connects isolated entities to the graph.
            logger.info("Computing co-occurrence edges from shared text chunks...")
            cooccur_result = session.run(
                "MATCH (a)-[:EXTRACTED_FROM]->(c:ChunkRef)<-[:EXTRACTED_FROM]-(b) "
                "WHERE a <> b AND NOT a:ChunkRef AND NOT b:ChunkRef "
                "AND id(a) < id(b) "  # avoid duplicates
                "RETURN coalesce(a.name, labels(a)[0]) AS source, "
                "coalesce(b.name, labels(b)[0]) AS target, "
                "count(c) AS shared_chunks, "
                "collect(c.chunk_id) AS chunk_ids "
                "ORDER BY shared_chunks DESC"
            )
            existing_pairs = {(r["source"], r["target"]) for r in rows}
            existing_pairs.update({(r["target"], r["source"]) for r in rows})
            cooccur_count = 0
            idx = len(rows)

            for record in cooccur_result:
                src = record["source"]
                tgt = record["target"]
                if (src, tgt) in existing_pairs:
                    continue
                shared = record["shared_chunks"]
                # Weight by co-occurrence strength (more shared chunks = stronger)
                weight = min(0.5 + shared * 0.1, 0.85)
                chunk_ids = [str(c) for c in record["chunk_ids"] if c]

                rows.append({
                    "id": str(uuid.uuid4()),
                    "source": src,
                    "target": tgt,
                    "description": f"{src} and {tgt} co-occur in {shared} text chunk(s)",
                    "weight": weight,
                    "human_readable_id": idx,
                    "text_unit_ids": chunk_ids,
                })
                existing_pairs.add((src, tgt))
                idx += 1
                cooccur_count += 1

            logger.info("Added %d co-occurrence edges", cooccur_count)

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
                "document_id": doc_id or "",
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

    # Sanitize text fields — some Unicode chars (non-breaking hyphen U+2011,
    # narrow no-break space U+202F) cause bge-m3 to produce NaN embeddings.
    _UNICODE_REPLACEMENTS = str.maketrans({
        "\u2011": "-",   # non-breaking hyphen → regular hyphen
        "\u2010": "-",   # hyphen → regular hyphen
        "\u2012": "-",   # figure dash
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u202f": " ",   # narrow no-break space
        "\u00a0": " ",   # no-break space
    })

    def _sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda v: v.translate(_UNICODE_REPLACEMENTS) if isinstance(v, str) else v
                )
        return df

    entities_df = _sanitize_text_columns(entities_df)
    relationships_df = _sanitize_text_columns(relationships_df)
    text_units_df = _sanitize_text_columns(text_units_df)

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
