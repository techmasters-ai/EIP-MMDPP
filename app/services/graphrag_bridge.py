"""Postgres -> GraphRAG input bridge layer.

Exports documents (full text) from Postgres into pandas DataFrames for
Microsoft GraphRAG's indexing pipeline. GraphRAG owns entity/relationship
extraction via its LLM pipeline.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Unicode chars that cause NaN in bge-m3 embeddings
_UNICODE_REPLACEMENTS = str.maketrans({
    "\u2011": "-",   # non-breaking hyphen
    "\u2010": "-",   # hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u202f": " ",   # narrow no-break space
    "\u00a0": " ",   # no-break space
})


def _sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Unicode chars that cause NaN in bge-m3 embeddings."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.translate(_UNICODE_REPLACEMENTS) if isinstance(v, str) else v
            )
    return df


def export_documents(db_session) -> pd.DataFrame:
    """Export full document text from Postgres for GraphRAG input.

    Concatenates all text chunks per document to reconstruct full document
    text. GraphRAG will re-chunk using its own chunker settings.
    """
    try:
        # Get document metadata
        doc_result = db_session.execute(text(
            "SELECT id::text, COALESCE(filename, 'untitled') AS title "
            "FROM ingest.documents"
        ))
        docs = {row[0]: {"id": row[0], "title": row[1]} for row in doc_result.fetchall()}

        # Get all text chunks ordered by position to reconstruct document text
        chunk_result = db_session.execute(text(
            "SELECT document_id::text, chunk_text "
            "FROM retrieval.text_chunks "
            "WHERE chunk_text IS NOT NULL "
            "ORDER BY document_id, id"
        ))

        doc_texts: dict[str, list[str]] = {}
        for row in chunk_result.fetchall():
            doc_id, chunk_text = row
            doc_texts.setdefault(doc_id, []).append(chunk_text or "")

        rows = []
        for doc_id, doc_info in docs.items():
            full_text = "\n\n".join(doc_texts.get(doc_id, []))
            # Append content hash to title so GraphRAG's delta comparison
            # detects modified documents (it compares by title only).
            content_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
            title = f"{doc_info['title']}_{content_hash}"
            rows.append({
                "id": doc_info["id"],
                "title": title,
                "text": full_text,
            })

    except Exception:
        logger.exception("Failed to export documents from Postgres")
        rows = []

    return pd.DataFrame(rows, columns=["id", "title", "text"])


def export_all(db_session, output_dir: Path) -> dict:
    """Export documents to input/ as CSV for GraphRAG indexing.

    GraphRAG's native input reader loads CSV files from input_storage.
    This lets load_update_documents run its own delta detection via
    get_delta_docs() (title-based comparison), halting the pipeline
    immediately when there are no new documents.

    Returns dict with counts of exported items.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_df = export_documents(db_session)
    documents_df = _sanitize_text_columns(documents_df)
    documents_df.to_csv(output_dir / "documents.csv", index=False)

    stats = {
        "documents": len(documents_df),
    }
    logger.info("GraphRAG bridge export complete: %s", stats)
    return stats
