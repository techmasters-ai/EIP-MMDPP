"""HybridChunker integration: post-process native chunks to substitute
normalized table chunks where appropriate.

See §10.1 of the design spec. The functions here are pure (no I/O);
they are invoked from app/workers/pipeline.py between the native
chunker call and the chunk-iteration loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.services.table_normalization.models import (
    NormalizedTable,
    Shape,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens
from app.services.table_normalization.render_graph import _render_column_as_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AdapterProv:
    page_no: int


@dataclass(frozen=True)
class _AdapterDocItem:
    self_ref: str
    prov: tuple[_AdapterProv, ...]


@dataclass(frozen=True)
class _AdapterMeta:
    doc_items: tuple[_AdapterDocItem, ...]
    headings: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedTableChunkAdapter:
    """Ducktypes the HybridChunker chunk interface.

    Read interface that pipeline.py:5559-5623 uses:
    - .text
    - .meta.doc_items[].self_ref
    - .meta.doc_items[].prov[].page_no
    - .meta.headings
    Plus .extra_metadata for chunk_metadata column population.

    CRITICAL: the single synthetic doc_item's self_ref is the TABLE-LEVEL
    ref ("#/tables/{N}"), never a cell ref. Cell refs flow through
    .extra_metadata.cell_refs only — keeps TextChunk.self_refs in today's
    shape (#/texts/N or #/tables/N), so provenance.py:_resolve_element_uid
    and the retrieval response surface stay backwards compatible.
    """
    etc: Any  # EmbeddingTableChunk or GraphTableChunk
    parent_headings: tuple[str, ...]
    parent_table_ref: str

    @property
    def text(self) -> str:
        return self.etc.text

    @property
    def meta(self) -> _AdapterMeta:
        prov_tuple = tuple(_AdapterProv(page_no=p) for p in self.etc.page_numbers)
        item = _AdapterDocItem(self_ref=self.parent_table_ref, prov=prov_tuple)
        return _AdapterMeta(doc_items=(item,), headings=self.parent_headings)

    @property
    def extra_metadata(self) -> dict:
        return {
            "chunk_kind": self.etc.chunk_kind.value,
            "table_ref": self.etc.table_ref,
            "entity_display_name": self.etc.entity_display_name,
            "section": self.etc.section,
            "column_index": self.etc.column_index,
            "cell_refs": list(self.etc.cell_refs),
            "row_labels": list(self.etc.row_labels),
            "page_numbers": list(self.etc.page_numbers),
        }


def _normalized_table_size_tokens(nt: NormalizedTable) -> int:
    """Canonical size function per spec rev. 7 §10.1.

    Sum of bge-m3 tokens across rendered columns. Single contract; no
    cheap fallback (boundary behavior at MIN_TABLE_NORMALIZATION_TOKENS
    must be deterministic).
    """
    return sum(
        count_bge_m3_tokens(_render_column_as_text(col, nt, nt.sections))
        for col in nt.columns
    )


def _classify_native_chunk(
    nc: Any, normalized_by_table_idx: dict[int, NormalizedTable],
) -> tuple[str, int | None]:
    """Classify a native chunk as table_dominant / table_mixed / non_table."""
    items = getattr(getattr(nc, "meta", None), "doc_items", None) or []
    if not items:
        return ("non_table", None)
    table_idx_counts: dict[int, int] = {}
    for item in items:
        ref = getattr(item, "self_ref", None) or ""
        if not ref.startswith("#/tables/"):
            continue
        try:
            idx = int(ref.split("/")[-1])
        except (ValueError, IndexError):
            continue
        if idx in normalized_by_table_idx and normalized_by_table_idx[idx].shape != Shape.OTHER:
            table_idx_counts[idx] = table_idx_counts.get(idx, 0) + 1
    if not table_idx_counts:
        return ("non_table", None)
    dominant_idx = max(table_idx_counts, key=table_idx_counts.get)
    dominant_share = table_idx_counts[dominant_idx] / len(items)
    return (("table_dominant" if dominant_share >= 0.8 else "table_mixed"), dominant_idx)


def _suppress_raw_table_texts(
    doc_json: dict,
    normalized: list[NormalizedTable],
) -> None:
    """Blank the flat-text mirrors of normalized non-OTHER tables in-place.

    Per §9.2 invariant:
    - len(doc_json['texts']) is UNCHANGED. No element is removed; no
      index shifts. This preserves self_ref stability for any code that
      references texts by index (children refs, prov entries, etc.).
    - doc_json['tables'] is NOT touched. The Phase 0/0.5 overlay
      machinery reads tables[] directly and must remain functional.
    - Tables with shape == OTHER keep their flat text (the OTHER
      fallback depends on it).
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    if not non_other:
        return
    target_refs = {f"#/tables/{i}" for i in non_other}
    for t in doc_json.get("texts") or []:
        if t.get("self_ref") in target_refs:
            t["text"] = ""
            t["orig"] = ""


def _substitute_table_chunks(
    native_chunks: list[Any],
    normalized_by_table_idx: dict[int, NormalizedTable],
    render_fn: Callable[..., list[Any]],
    *,
    token_limit: int,
    summary_limit: int,
    min_table_tokens: int,
) -> list[Any]:
    """Per §10.1 of the design spec.

    Substitution decision tree:
    - non_table: pass through unchanged.
    - normalized table below min_table_tokens: pass through unchanged.
    - table_dominant: substitute entirely. Subsequent natives for the same
      table_idx are dropped (NormalizedTable.cells covers all content).
    - table_mixed above threshold: emit normalized chunks AND keep native
      (defensive — wide tables shouldn't reach this branch per spike findings).
    """
    seen_table_idx: set[int] = set()
    out: list[Any] = []
    for nc in native_chunks:
        cls, table_idx = _classify_native_chunk(nc, normalized_by_table_idx)

        if cls == "non_table":
            out.append(nc)
            continue

        nt = normalized_by_table_idx[table_idx]
        size = _normalized_table_size_tokens(nt)
        if size < min_table_tokens:
            out.append(nc)
            continue

        parent_headings = tuple(getattr(getattr(nc, "meta", None), "headings", None) or [])
        parent_table_ref = f"#/tables/{table_idx}"

        if cls == "table_dominant":
            if table_idx in seen_table_idx:
                continue
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
            continue

        # cls == "table_mixed"
        logger.warning(
            "_substitute_table_chunks: table_mixed classification fired (table_idx=%d, "
            "dominant_share<0.8). HybridChunker merged this table with prose; "
            "emitting normalized chunks AND keeping native (degraded path).",
            table_idx,
        )
        if table_idx not in seen_table_idx:
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
        out.append(nc)
    return out
