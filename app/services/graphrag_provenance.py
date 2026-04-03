"""Build provenance data from GraphRAG search context.

Organizes entities, relationships, and covariates under their community
reports with source document traceability.  No LLM cooperation needed --
uses the context DataFrames that GraphRAG already returns.

Provenance schema (per report):
  report_id, report_title, report_content
  entities[]   – id, entity, description, number of relationships, in_context,
                 document_name, document_id, original_text
  relationships[] – id, source, target, description, weight, links, in_context,
                    document_name, document_id, original_text
  covariates[]    – id, description, …, document_name, document_id, original_text
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_RE_TITLE_HASH = re.compile(r"_[0-9a-f]{8}$")
_RE_DATA_BLOCK = re.compile(r"\[Data:\s*([^\]]+)\]", re.IGNORECASE)
_RE_TYPE_IDS = re.compile(
    r"(Entities|Relationships|Sources)\s*\(([^)]+)\)", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_value(val: Any) -> Any:
    """Convert a single pandas/numpy value to a JSON-safe Python type."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "item"):  # numpy scalar → native Python
        return val.item()
    return val


def _parse_report_data_references(content: str) -> dict[str, set[int]]:
    """Parse ``[Data: Entities/Relationships/Sources (...)]`` from report content.

    Handles single-type blocks like ``[Data: Entities (1, 2)]`` and combined
    blocks like ``[Data: Sources (2, 13); Entities (6, 7)]``.

    Returns dict mapping ``'entities'``, ``'relationships'``, ``'sources'``
    to sets of human-readable IDs referenced in the text.
    """
    refs: dict[str, set[int]] = {
        "entities": set(),
        "relationships": set(),
        "sources": set(),
    }
    for block in _RE_DATA_BLOCK.finditer(content):
        for m in _RE_TYPE_IDS.finditer(block.group(1)):
            key = m.group(1).lower()
            for tok in m.group(2).split(","):
                tok = tok.strip()
                if tok.isdigit():
                    refs[key].add(int(tok))
    return refs


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


def _resolve_source_info(
    text_unit_ids: list | None,
    tu_df: pd.DataFrame,
    doc_df: pd.DataFrame,
) -> dict[str, str]:
    """Resolve text_unit_ids (UUIDs) → document_name, document_id, original_text.

    Returns the first successfully resolved text unit's data.
    """
    empty = {"document_name": "", "document_id": "", "original_text": ""}
    if not text_unit_ids or tu_df.empty:
        return empty
    for tu_id in text_unit_ids:
        if not tu_id or (isinstance(tu_id, float) and pd.isna(tu_id)):
            continue
        rows = tu_df[tu_df["id"] == tu_id]
        if rows.empty:
            continue
        tu_row = rows.iloc[0]
        original_text = str(tu_row.get("text", ""))
        doc_id = tu_row.get("document_id")
        doc_name = ""
        if doc_id and not (isinstance(doc_id, float) and pd.isna(doc_id)):
            if not doc_df.empty:
                doc_rows = doc_df[doc_df["id"] == doc_id]
                if not doc_rows.empty:
                    doc_name = _clean_doc_title(str(doc_rows.iloc[0].get("title", "")))
            return {
                "document_name": doc_name,
                "document_id": str(doc_id),
                "original_text": original_text,
            }
        return {
            "document_name": "",
            "document_id": "",
            "original_text": original_text,
        }
    return empty


def _build_hrid_to_uuid_map(parquet_df: pd.DataFrame) -> dict[int, str]:
    """Build a mapping from human_readable_id -> UUID id for a Parquet DataFrame."""
    if parquet_df.empty or "human_readable_id" not in parquet_df.columns:
        return {}
    return dict(zip(parquet_df["human_readable_id"], parquet_df["id"]))


def _include_extra_columns(
    base: dict[str, Any], row: pd.Series, exclude: set[str],
) -> None:
    """Add any context DataFrame columns not already in *base*."""
    for col in row.index:
        if col not in base and col not in exclude:
            base[col] = _safe_value(row[col])


def _ensure_df(value) -> pd.DataFrame:
    """Coerce a context value to a DataFrame.

    GraphRAG context dicts may contain DataFrames or plain lists (e.g. empty
    ``[]``).  This normalises both forms so callers can safely use ``.empty``,
    ``.iterrows()``, etc.
    """
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return pd.DataFrame(value) if value else pd.DataFrame()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-report provenance builder
# ---------------------------------------------------------------------------

def _build_report_provenance(
    report_row: pd.Series,
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    community_entity_uuids: set[str],
    community_rel_uuids: set[str],
    community_tu_uuids: set[str],
    *,
    entity_fallback: bool = True,
) -> dict[str, Any]:
    """Build a single provenance entry for one community report.

    Uses a three-tier filtering strategy for each context item type:
      1. Community UUID membership (from communities.parquet)
      2. ``[Data: ...]`` references parsed from report content
      3. Include all context items (no filter)

    All columns present in the context DataFrames are carried through, plus
    ``document_name``, ``document_id``, and ``original_text`` resolved from
    the Parquet text-unit / document tables.
    """
    tu_df = data.get("text_units", pd.DataFrame())
    doc_df = data.get("documents", pd.DataFrame())
    ent_parquet = data.get("entities", pd.DataFrame())
    rel_parquet = data.get("relationships", pd.DataFrame())

    ent_hrid_to_uuid = _build_hrid_to_uuid_map(ent_parquet)
    rel_hrid_to_uuid = _build_hrid_to_uuid_map(rel_parquet)

    report_content = str(
        report_row.get("content", report_row.get("full_content", ""))
    )

    entry: dict[str, Any] = {
        "report_id": str(report_row.get("id", "")),
        "report_title": str(report_row.get("title", "")),
        "report_content": report_content,
        "entities": [],
        "relationships": [],
        "covariates": [],
    }

    # Fallback: parse [Data: ...] references from report content
    refs = _parse_report_data_references(report_content)

    # --- Entities ---
    ctx_entities = _ensure_df(context.get("entities", pd.DataFrame()))
    if not ctx_entities.empty:
        for _, row in ctx_entities.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            hrid_int = int(hrid)

            # Three-tier filter
            if community_entity_uuids:
                uuid = ent_hrid_to_uuid.get(hrid_int)
                if not (uuid and uuid in community_entity_uuids):
                    continue
            elif refs["entities"]:
                if hrid_int not in refs["entities"]:
                    continue
            # else: include all

            # Resolve source document + original text via Parquet
            uuid = ent_hrid_to_uuid.get(hrid_int)
            tu_ids = None
            ent_type = ""
            if uuid and not ent_parquet.empty:
                parquet_rows = ent_parquet[ent_parquet["id"] == uuid]
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                    ent_type = str(parquet_rows.iloc[0].get("type", ""))
            elif not uuid:
                logger.debug(
                    "Provenance: no UUID mapping for entity HRID %d", hrid_int,
                )

            source = _resolve_source_info(
                tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
            )

            ent_entry: dict[str, Any] = {
                "id": hrid_int,
                "entity": str(row.get("entity", row.get("title", ""))),
                "description": str(row.get("description", "")),
                "type": ent_type,
                "document_name": source["document_name"],
                "document_id": source["document_id"],
                "original_text": source["original_text"],
            }
            _include_extra_columns(ent_entry, row, {"id", "entity", "title"})
            entry["entities"].append(ent_entry)

    # Parquet fallback: when entities are still empty, load directly from
    # Parquet via community membership or report content references.
    # Skipped for global search (entity_fallback=False).
    if entity_fallback and not entry["entities"] and not ent_parquet.empty:
        uuids_to_load: set[str] = set()
        if community_entity_uuids:
            uuids_to_load = community_entity_uuids
        elif refs["entities"]:
            uuids_to_load = {
                ent_hrid_to_uuid[h]
                for h in refs["entities"]
                if h in ent_hrid_to_uuid
            }
        if uuids_to_load:
            logger.info(
                "Provenance: context has no entities; loading %d from Parquet",
                len(uuids_to_load),
            )
        for uuid in uuids_to_load:
            parquet_rows = ent_parquet[ent_parquet["id"] == uuid]
            if parquet_rows.empty:
                continue
            prow = parquet_rows.iloc[0]
            tu_ids = prow.get("text_unit_ids")
            source = _resolve_source_info(
                tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
            )
            entry["entities"].append({
                "id": int(prow.get("human_readable_id", 0)),
                "entity": str(prow.get("title", "")),
                "description": str(prow.get("description", "")),
                "type": str(prow.get("type", "")),
                "document_name": source["document_name"],
                "document_id": source["document_id"],
                "original_text": source["original_text"],
            })

    # --- Relationships ---
    ctx_rels = _ensure_df(context.get("relationships", pd.DataFrame()))
    if not ctx_rels.empty:
        for _, row in ctx_rels.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            hrid_int = int(hrid)

            if community_rel_uuids:
                uuid = rel_hrid_to_uuid.get(hrid_int)
                if not (uuid and uuid in community_rel_uuids):
                    continue
            elif refs["relationships"]:
                if hrid_int not in refs["relationships"]:
                    continue

            uuid = rel_hrid_to_uuid.get(hrid_int)
            tu_ids = None
            if not uuid:
                logger.debug(
                    "Provenance: no UUID mapping for relationship HRID %d",
                    hrid_int,
                )
            elif not rel_parquet.empty:
                parquet_rows = rel_parquet[rel_parquet["id"] == uuid]
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                    if not tu_ids:
                        logger.debug(
                            "Provenance: relationship %s (HRID %d) has no text_unit_ids",
                            uuid, hrid_int,
                        )

            source = _resolve_source_info(
                tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
            )

            rel_entry: dict[str, Any] = {
                "id": hrid_int,
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "description": str(row.get("description", "")),
                "document_name": source["document_name"],
                "document_id": source["document_id"],
                "original_text": source["original_text"],
            }
            _include_extra_columns(rel_entry, row, {"id"})
            entry["relationships"].append(rel_entry)

    # --- Covariates (keyed as "claims" in context) ---
    ctx_claims = _ensure_df(context.get("claims", pd.DataFrame()))
    if not ctx_claims.empty:
        cov_parquet = data.get("covariates", pd.DataFrame())
        cov_hrid_to_uuid = _build_hrid_to_uuid_map(cov_parquet)
        for _, row in ctx_claims.iterrows():
            cov_hrid = row.get("id")
            tu_ids = None
            if cov_hrid is not None and not cov_parquet.empty:
                uuid = cov_hrid_to_uuid.get(int(cov_hrid))
                if uuid:
                    parquet_rows = cov_parquet[cov_parquet["id"] == uuid]
                    if not parquet_rows.empty:
                        tu_ids = parquet_rows.iloc[0].get("text_unit_ids")

            source = _resolve_source_info(
                tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
            )

            cov_entry: dict[str, Any] = {
                "id": int(cov_hrid) if cov_hrid is not None else 0,
                "description": str(row.get("description", "")),
                "document_name": source["document_name"],
                "document_id": source["document_id"],
                "original_text": source["original_text"],
            }
            _include_extra_columns(cov_entry, row, {"id"})
            entry["covariates"].append(cov_entry)

    return entry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_provenance(
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict[str, Any]]:
    """Build provenance from GraphRAG search context.

    Organizes entities, relationships, and covariates under community reports.
    Each item includes document_name, document_id, and original_text.

    Args:
        context: context_records dict from GraphRAG search. Keys vary by strategy.
        data: full Parquet data from _load_search_data().
        strategy: search strategy string (graphrag_local, graphrag_global, etc.).

    Returns:
        List of provenance entries, one per community report.
    """
    ctx_reports = _ensure_df(context.get("reports", pd.DataFrame()))
    cr_parquet = data.get("community_reports", pd.DataFrame())
    comm_parquet = data.get("communities", pd.DataFrame())

    # Basic search: no reports, just text units
    if strategy in ("graphrag_basic", "basic"):
        ctx_sources = _ensure_df(context.get("sources", pd.DataFrame()))
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
            text_units.append({
                "id": int(hrid) if hrid is not None else 0,
                "text": str(row.get("text", "")),
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

        use_entity_fallback = strategy not in ("graphrag_global", "global")
        entry = _build_report_provenance(
            report_row, context, data,
            community_entity_uuids, community_rel_uuids, community_tu_uuids,
            entity_fallback=use_entity_fallback,
        )
        provenance.append(entry)

    return provenance
