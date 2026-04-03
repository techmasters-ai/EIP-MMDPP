"""Build provenance data from GraphRAG search context.

Organizes entities, relationships, text units, and covariates under their
community reports with source document traceability. No LLM cooperation
needed — uses the context DataFrames that GraphRAG already returns.
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_RE_TITLE_HASH = re.compile(r"_[0-9a-f]{8}$")


def _clean_doc_title(title: str) -> str:
    """Strip the content hash suffix from bridge-layer document titles."""
    return _RE_TITLE_HASH.sub("", title)


def _resolve_doc(doc_id: str | None, doc_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Resolve a single document_id to a source_documents entry."""
    if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
        return []
    if doc_df.empty:
        return [{"document_id": doc_id, "document_title": ""}]
    rows = doc_df[doc_df["id"] == doc_id]
    if rows.empty:
        return [{"document_id": doc_id, "document_title": ""}]
    title = _clean_doc_title(str(rows.iloc[0].get("title", "")))
    return [{"document_id": doc_id, "document_title": title}]


def _resolve_docs_via_text_units(
    text_unit_ids: list | None,
    tu_df: pd.DataFrame,
    doc_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Resolve text_unit_ids (UUIDs) to source documents, deduplicated."""
    if not text_unit_ids or tu_df.empty:
        return []
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tu_id in text_unit_ids:
        if not tu_id or (isinstance(tu_id, float) and pd.isna(tu_id)):
            continue
        rows = tu_df[tu_df["id"] == tu_id]
        if rows.empty:
            continue
        doc_id = rows.iloc[0].get("document_id")
        if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.extend(_resolve_doc(doc_id, doc_df))
    return docs


def _build_hrid_to_uuid_map(parquet_df: pd.DataFrame) -> dict[int, str]:
    """Build a mapping from human_readable_id -> UUID id for a Parquet DataFrame."""
    if parquet_df.empty or "human_readable_id" not in parquet_df.columns:
        return {}
    return dict(zip(parquet_df["human_readable_id"], parquet_df["id"]))


def _build_report_provenance(
    report_row: pd.Series,
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    community_entity_uuids: set[str],
    community_rel_uuids: set[str],
    community_tu_uuids: set[str],
) -> dict[str, Any]:
    """Build a single provenance entry for one community report."""
    tu_df = data.get("text_units", pd.DataFrame())
    doc_df = data.get("documents", pd.DataFrame())
    ent_parquet = data.get("entities", pd.DataFrame())
    rel_parquet = data.get("relationships", pd.DataFrame())

    ent_hrid_to_uuid = _build_hrid_to_uuid_map(ent_parquet)
    rel_hrid_to_uuid = _build_hrid_to_uuid_map(rel_parquet)
    tu_hrid_to_uuid = _build_hrid_to_uuid_map(tu_df)

    entry: dict[str, Any] = {
        "report_id": str(report_row.get("id", "")),
        "report_title": str(report_row.get("title", "")),
        "report_content": str(report_row.get("content", report_row.get("full_content", ""))),
        "entities": [],
        "relationships": [],
        "text_units": [],
        "covariates": [],
    }

    # Filter context entities to this community
    ctx_entities = context.get("entities", pd.DataFrame())
    if not ctx_entities.empty and community_entity_uuids:
        for _, row in ctx_entities.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = ent_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_entity_uuids:
                parquet_rows = ent_parquet[ent_parquet["id"] == uuid]
                tu_ids = None
                ent_type = ""
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                    ent_type = str(parquet_rows.iloc[0].get("type", ""))
                entry["entities"].append({
                    "id": int(hrid),
                    "title": str(row.get("entity", row.get("title", ""))),
                    "type": ent_type,
                    "description": str(row.get("description", "")),
                    "source_documents": _resolve_docs_via_text_units(
                        tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
                    ),
                })

    # Filter context relationships to this community
    ctx_rels = context.get("relationships", pd.DataFrame())
    if not ctx_rels.empty and community_rel_uuids:
        for _, row in ctx_rels.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = rel_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_rel_uuids:
                parquet_rows = rel_parquet[rel_parquet["id"] == uuid]
                tu_ids = None
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                entry["relationships"].append({
                    "id": int(hrid),
                    "source": str(row.get("source", "")),
                    "target": str(row.get("target", "")),
                    "description": str(row.get("description", "")),
                    "source_documents": _resolve_docs_via_text_units(
                        tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
                    ),
                })

    # Filter context text_units (keyed as "sources") to this community
    ctx_sources = context.get("sources", pd.DataFrame())
    if not ctx_sources.empty and community_tu_uuids:
        for _, row in ctx_sources.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = tu_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_tu_uuids:
                tu_parquet_rows = tu_df[tu_df["id"] == uuid]
                doc_id = None
                if not tu_parquet_rows.empty:
                    doc_id = tu_parquet_rows.iloc[0].get("document_id")
                text = str(row.get("text", ""))
                if len(text) > 500:
                    text = text[:500] + "..."
                entry["text_units"].append({
                    "id": int(hrid),
                    "text": text,
                    "source_documents": _resolve_doc(doc_id, doc_df),
                })

    # Filter context covariates (keyed as "claims") to this community
    ctx_claims = context.get("claims", pd.DataFrame())
    if not ctx_claims.empty:
        for _, row in ctx_claims.iterrows():
            entry["covariates"].append({
                "id": int(row.get("id", 0)),
                "description": str(row.get("description", "")),
                "source_documents": [],
            })

    return entry


def build_provenance(
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict[str, Any]]:
    """Build provenance from GraphRAG search context.

    Organizes entities, relationships, text units under community reports.
    Each item includes source document traceability.

    Args:
        context: context_records dict from GraphRAG search. Keys vary by strategy.
        data: full Parquet data from _load_search_data().
        strategy: search strategy string (graphrag_local, graphrag_global, etc.).

    Returns:
        List of provenance entries, one per community report.
    """
    ctx_reports = context.get("reports", pd.DataFrame())
    cr_parquet = data.get("community_reports", pd.DataFrame())
    comm_parquet = data.get("communities", pd.DataFrame())

    # Basic search: no reports, just text units
    if strategy in ("graphrag_basic", "basic"):
        ctx_sources = context.get("sources", pd.DataFrame())
        if ctx_sources.empty:
            return []
        tu_df = data.get("text_units", pd.DataFrame())
        doc_df = data.get("documents", pd.DataFrame())
        tu_hrid_to_uuid = _build_hrid_to_uuid_map(tu_df)
        text_units = []
        for _, row in ctx_sources.iterrows():
            hrid = row.get("id")
            uuid = tu_hrid_to_uuid.get(int(hrid)) if hrid is not None else None
            doc_id = None
            if uuid and not tu_df.empty:
                parquet_rows = tu_df[tu_df["id"] == uuid]
                if not parquet_rows.empty:
                    doc_id = parquet_rows.iloc[0].get("document_id")
            text = str(row.get("text", ""))
            if len(text) > 500:
                text = text[:500] + "..."
            text_units.append({
                "id": int(hrid) if hrid is not None else 0,
                "text": text,
                "source_documents": _resolve_doc(doc_id, doc_df),
            })
        return [{
            "report_id": None,
            "report_title": None,
            "report_content": None,
            "entities": [],
            "relationships": [],
            "text_units": text_units,
            "covariates": [],
        }]

    # No reports in context -> empty provenance
    if ctx_reports.empty:
        return []

    provenance: list[dict[str, Any]] = []

    for _, report_row in ctx_reports.iterrows():
        report_hrid = report_row.get("id")

        community_entity_uuids: set[str] = set()
        community_rel_uuids: set[str] = set()
        community_tu_uuids: set[str] = set()

        if report_hrid is not None and not cr_parquet.empty:
            cr_match = cr_parquet[cr_parquet["human_readable_id"] == int(report_hrid)]
            if not cr_match.empty:
                community_id = cr_match.iloc[0].get("community")
                if community_id is not None and not comm_parquet.empty:
                    comm_match = comm_parquet[comm_parquet["community"] == community_id]
                    if not comm_match.empty:
                        comm_row = comm_match.iloc[0]
                        ent_ids = comm_row.get("entity_ids")
                        if isinstance(ent_ids, list):
                            community_entity_uuids = set(ent_ids)
                        rel_ids = comm_row.get("relationship_ids")
                        if isinstance(rel_ids, list):
                            community_rel_uuids = set(rel_ids)
                        tu_ids = comm_row.get("text_unit_ids")
                        if isinstance(tu_ids, list):
                            community_tu_uuids = set(tu_ids)
            else:
                logger.warning(
                    "Provenance: report hrid=%s not found in community_reports Parquet",
                    report_hrid,
                )

        entry = _build_report_provenance(
            report_row, context, data,
            community_entity_uuids, community_rel_uuids, community_tu_uuids,
        )
        provenance.append(entry)

    return provenance
