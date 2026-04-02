"""Parse and resolve inline citations from GraphRAG LLM responses.

Supports two citation styles:

1. **Our prompted format** — `## Sources` block with `[n]` markers:
   - ID-based (Local/Drift): [n] Entity: NAME (human_readable_id), Relationship: id
   - Name-based (Global): [n] Entity: NAME
   - Text-based (Basic): [n] Source: "text excerpt..."

2. **GraphRAG native format** — inline `【Data: ...】` markers produced by the
   graphrag library's own prompts:
   - 【Data: Sources (348, 349, 351)】  — text_unit human_readable_ids
   - 【Data: Entities (3293, 416)】     — entity human_readable_ids
   - 【Data: Relationships (5628)】     — relationship human_readable_ids
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_RE_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_RE_SOURCES_BLOCK = re.compile(r"\n##\s*Sources\s*\n(.*)", re.DOTALL)
_RE_CITATION_NUM = re.compile(r"^\[(\d+)\]\s*(.*)", re.MULTILINE)

# ID-based: [1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276
_RE_ENTITY_ID = re.compile(r"Entity:\s*(.+?)\s*\((\d+)\)")
_RE_REL_ID = re.compile(r"Relationship:\s*(\d+)")

# Name-based: [1] Entity: SA-2 GUIDELINE
_RE_ENTITY_NAME = re.compile(r"Entity:\s*(.+?)(?:,|$)")

# Text-based: [1] Source: "text excerpt..."
_RE_SOURCE_TEXT = re.compile(r'Source:\s*"(.+?)"')

# Document title hash suffix: "Title_a3b2c1d4" -> "Title"
_RE_TITLE_HASH = re.compile(r"_[0-9a-f]{8}$")

# GraphRAG native format: 【Data: Sources (348, 349, 351, 352, 368)】
_RE_NATIVE_MARKER = re.compile(
    r"【Data:\s*(Sources|Entities|Relationships)\s*\(([^)]+)\)】"
)


def strip_sources_block(response_text: str) -> tuple[str, str]:
    """Strip the ## Sources block and <think> tags from response text.

    Returns (clean_text, sources_block_text).
    """
    text = _RE_THINK_TAGS.sub("", response_text).strip()
    match = _RE_SOURCES_BLOCK.search(text)
    if not match:
        return text, ""
    clean = text[: match.start()].rstrip()
    block = match.group(1).strip()
    return clean, block


def parse_citation_block(
    block: str, strategy: str,
) -> dict[int, dict[str, Any]]:
    """Parse the sources block into a dict keyed by citation number.

    Strategy determines the parsing format:
    - local/drift: ID-based (entity IDs + relationship IDs)
    - global: name-based (entity names)
    - basic: text-based (source text excerpts)
    """
    citations: dict[int, dict[str, Any]] = {}

    for match in _RE_CITATION_NUM.finditer(block):
        num = int(match.group(1))
        if num in citations:
            continue  # keep first occurrence
        line = match.group(2)

        if strategy in ("graphrag_local", "graphrag_drift", "local", "drift"):
            entity_ids = [int(m.group(2)) for m in _RE_ENTITY_ID.finditer(line)]
            rel_ids = [int(m.group(1)) for m in _RE_REL_ID.finditer(line)]
            citations[num] = {"entity_ids": entity_ids, "relationship_ids": rel_ids}

        elif strategy in ("graphrag_global", "global"):
            names = [m.group(1).strip() for m in _RE_ENTITY_NAME.finditer(line)]
            citations[num] = {"entity_names": names}

        elif strategy in ("graphrag_basic", "basic"):
            text_match = _RE_SOURCE_TEXT.search(line)
            citations[num] = {
                "source_text": text_match.group(1) if text_match else line,
            }
        else:
            logger.debug("Unknown strategy %s for citation line: %s", strategy, line)

    return citations


def _clean_doc_title(title: str) -> str:
    """Strip the content hash suffix from bridge-layer document titles."""
    return _RE_TITLE_HASH.sub("", title)


def _resolve_text_unit_docs(
    text_unit_ids: list[str],
    data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Resolve text_unit_ids to source documents."""
    tu_df = data.get("text_units", pd.DataFrame())
    doc_df = data.get("documents", pd.DataFrame())
    if tu_df.empty:
        return []

    docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for tu_id in text_unit_ids:
        rows = tu_df[tu_df["id"] == tu_id]
        if rows.empty:
            continue
        row = rows.iloc[0]
        # document_ids is a list column in the Parquet schema
        doc_ids_raw = row.get("document_ids")
        if doc_ids_raw is None or (hasattr(doc_ids_raw, "__len__") and len(doc_ids_raw) == 0):
            continue
        doc_id_list = doc_ids_raw if hasattr(doc_ids_raw, "__iter__") and not isinstance(doc_ids_raw, str) else [doc_ids_raw]

        source_text = str(row.get("text", ""))
        if len(source_text) > 500:
            source_text = source_text[:500] + "..."

        for doc_id in doc_id_list:
            if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
                continue
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            doc_title = ""
            if not doc_df.empty:
                doc_rows = doc_df[doc_df["id"] == doc_id]
                if not doc_rows.empty:
                    doc_title = _clean_doc_title(str(doc_rows.iloc[0].get("title", "")))

            docs.append({
                "document_id": doc_id,
                "document_title": doc_title,
                "source_text": source_text,
            })

    return docs


def resolve_citations(
    parsed: dict[int, dict[str, Any]],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict[str, Any]]:
    """Resolve parsed citations to full provenance data.

    Returns a list of source entries, one per citation number.
    """
    ent_df = data.get("entities", pd.DataFrame())
    rel_df = data.get("relationships", pd.DataFrame())
    sources: list[dict[str, Any]] = []

    for num in sorted(parsed.keys()):
        citation = parsed[num]
        entry: dict[str, Any] = {
            "citation": num,
            "entities": [],
            "relationships": [],
            "source_documents": [],
        }
        all_tu_ids: list[str] = []

        # Resolve entities
        if "entity_ids" in citation:
            for eid in citation["entity_ids"]:
                rows = ent_df[ent_df["human_readable_id"] == eid] if not ent_df.empty else pd.DataFrame()
                if rows.empty:
                    logger.warning("Citation [%d]: entity ID %d not found", num, eid)
                    continue
                row = rows.iloc[0]
                entry["entities"].append({
                    "id": int(eid),
                    "title": str(row.get("title", "")),
                    "type": str(row.get("type", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        elif "entity_names" in citation:
            for name in citation["entity_names"]:
                if ent_df.empty:
                    continue
                rows = ent_df[ent_df["title"].str.upper() == name.upper()]
                if rows.empty:
                    logger.warning("Citation [%d]: entity name '%s' not found", num, name)
                    continue
                row = rows.iloc[0]
                entry["entities"].append({
                    "id": int(row.get("human_readable_id", 0)),
                    "title": str(row.get("title", "")),
                    "type": str(row.get("type", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        elif "source_text" in citation:
            # Basic search: match text units by substring
            tu_df = data.get("text_units", pd.DataFrame())
            if not tu_df.empty:
                excerpt = citation["source_text"][:100]
                matches = tu_df[tu_df["text"].str.contains(excerpt, case=False, na=False)]
                if not matches.empty:
                    all_tu_ids.extend(matches["id"].tolist())
                else:
                    logger.warning("Citation [%d]: text excerpt not matched", num)

        # Resolve relationships
        if "relationship_ids" in citation:
            for rid in citation["relationship_ids"]:
                rows = rel_df[rel_df["human_readable_id"] == rid] if not rel_df.empty else pd.DataFrame()
                if rows.empty:
                    logger.warning("Citation [%d]: relationship ID %d not found", num, rid)
                    continue
                row = rows.iloc[0]
                entry["relationships"].append({
                    "id": int(rid),
                    "source": str(row.get("source", "")),
                    "target": str(row.get("target", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        # Resolve source documents from collected text_unit_ids
        entry["source_documents"] = _resolve_text_unit_docs(all_tu_ids, data)

        sources.append(entry)

    return sources


def _parse_native_markers(
    response_text: str,
) -> tuple[str, dict[int, dict[str, Any]]]:
    """Parse GraphRAG native 【Data: ...】 markers into numbered citations.

    Each unique combination of IDs within a marker type becomes a citation.
    The markers are replaced with [n] in the returned clean text.

    Returns (clean_text, parsed_citations).
    """
    # Collect all unique markers in order of first appearance
    marker_to_num: dict[str, int] = {}
    citations: dict[int, dict[str, Any]] = {}
    counter = 1

    for match in _RE_NATIVE_MARKER.finditer(response_text):
        kind = match.group(1)  # Sources, Entities, or Relationships
        ids_str = match.group(2)  # "348, 349, 351"
        ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
        if not ids:
            continue

        # Use the full marker text as dedup key
        marker_key = match.group(0)
        if marker_key not in marker_to_num:
            marker_to_num[marker_key] = counter
            num = counter
            counter += 1

            entry: dict[str, Any] = {}
            if kind == "Sources":
                entry["text_unit_ids"] = ids
            elif kind == "Entities":
                entry["entity_ids"] = ids
            elif kind == "Relationships":
                entry["relationship_ids"] = ids
            citations[num] = entry
        # else: reuse existing citation number

    if not citations:
        return response_text, {}

    # Replace markers with [n] in text
    def _replace(m: re.Match) -> str:
        key = m.group(0)
        num = marker_to_num.get(key)
        return f"[{num}]" if num else ""

    clean = _RE_NATIVE_MARKER.sub(_replace, response_text)
    return clean, citations


def _resolve_native_citations(
    parsed: dict[int, dict[str, Any]],
    data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Resolve native-format citations (text_unit_ids, entity_ids, relationship_ids).

    These use human_readable_id values from the Parquet data.
    """
    ent_df = data.get("entities", pd.DataFrame())
    rel_df = data.get("relationships", pd.DataFrame())
    tu_df = data.get("text_units", pd.DataFrame())
    sources: list[dict[str, Any]] = []

    for num in sorted(parsed.keys()):
        citation = parsed[num]
        entry: dict[str, Any] = {
            "citation": num,
            "entities": [],
            "relationships": [],
            "source_documents": [],
        }
        all_tu_ids: list[str] = []

        # Resolve entity human_readable_ids
        for eid in citation.get("entity_ids", []):
            if ent_df.empty:
                continue
            rows = ent_df[ent_df["human_readable_id"] == eid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            entry["entities"].append({
                "id": int(eid),
                "title": str(row.get("title", "")),
                "type": str(row.get("type", "")),
                "description": str(row.get("description", "")),
            })
            tu_ids = row.get("text_unit_ids")
            if tu_ids is not None:
                all_tu_ids.extend(
                    tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else []
                )

        # Resolve relationship human_readable_ids
        for rid in citation.get("relationship_ids", []):
            if rel_df.empty:
                continue
            rows = rel_df[rel_df["human_readable_id"] == rid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            entry["relationships"].append({
                "id": int(rid),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "description": str(row.get("description", "")),
            })
            tu_ids = row.get("text_unit_ids")
            if tu_ids is not None:
                all_tu_ids.extend(
                    tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else []
                )

        # Resolve text_unit human_readable_ids directly to source documents
        for tu_hrid in citation.get("text_unit_ids", []):
            if tu_df.empty:
                continue
            rows = tu_df[tu_df["human_readable_id"] == tu_hrid]
            if rows.empty:
                continue
            all_tu_ids.append(str(rows.iloc[0]["id"]))

        entry["source_documents"] = _resolve_text_unit_docs(all_tu_ids, data)
        sources.append(entry)

    return sources


def process_citations(
    response_text: str,
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Top-level function: strip, parse, resolve citations.

    Tries our prompted ## Sources format first. Falls back to parsing
    GraphRAG's native 【Data: Sources/Entities/Relationships (...)】 markers.

    Returns (clean_response_text, sources_array).
    """
    # Strip think tags first
    text = _RE_THINK_TAGS.sub("", response_text).strip()

    # Try our prompted format first
    clean, block = strip_sources_block(text)
    if block:
        parsed = parse_citation_block(block, strategy)
        if parsed:
            sources = resolve_citations(parsed, data, strategy)
            return clean, sources

    # Fallback: parse GraphRAG native 【Data: ...】 markers
    if _RE_NATIVE_MARKER.search(text):
        clean_native, parsed_native = _parse_native_markers(text)
        if parsed_native:
            sources = _resolve_native_citations(parsed_native, data)
            return clean_native, sources

    return text, []
