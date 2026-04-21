"""Docling anchor walker — deterministic document structure emission.

Replaces the LLM reference pass (deleted in C-1) for SECTION / FIGURE /
TABLE / DOCUMENT entities. Structure is derived from the DoclingDocument
tree, not the LLM, per docs R-rules and spec §3.3.

This module exposes:
  * ``_extract_document_number_from_front_matter`` — scans the first
    N titles/section-headers for a MIL-STD / TM / ISO-style designator
    (implemented in D-2).
  * ``walk`` — traverses a DoclingDocument and emits MergedEntityRecord
    + MergedEdgeRecord instances for the anchor ontology (D-3).
"""
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableItem

from app.services.extraction_merge import (
    MergedEdgeRecord,
    MergedExtraction,
    _build_logical_identity,
    _to_merged_entity_record,
)
from ontology_bundles.air_defense_v3.entities import (
    DocumentEntity,
    FigureEntity,
    SectionEntity,
    TableEntity,
)

if TYPE_CHECKING:
    pass


# Matches MIL-STD/TM/MIL-DTL/MIL-HDBK/MIL-PRF/ANSI-IEEE/ISO/DoD style
# designators anywhere in a title or heading. Requires the designator
# token to be followed by at least one alphanumeric identifier char so
# plain text like "See TM" alone doesn't match.
_DOC_NUMBER_RE = re.compile(
    r"\b(?:TM|MIL-STD|MIL-DTL|MIL-HDBK|MIL-PRF|ANSI/IEEE|ISO|DoD)"
    r"\s*[-\s]?[A-Z0-9][\w.-]+",
    re.IGNORECASE,
)

_FRONT_MATTER_LIMIT = 30


def _extract_document_number_from_front_matter(
    docling_doc: DoclingDocument,
) -> str | None:
    """Return the first MIL-STD/TM/ISO-style designator found in the
    first ``_FRONT_MATTER_LIMIT`` title or section-header items, or
    None if none is found.

    Case-insensitive; the matched substring is returned with its
    original casing. Non-title / non-section-header items advance the
    30-item limit counter but are not scanned for designators — docs
    R-rule: document number lives in front matter, not body.
    """
    count = 0
    for item, _level in docling_doc.iterate_items():
        if count >= _FRONT_MATTER_LIMIT:
            return None
        count += 1
        label = getattr(item, "label", None)
        if label not in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            continue
        text = getattr(item, "text", None) or ""
        match = _DOC_NUMBER_RE.search(text)
        if match:
            return match.group(0).strip()
    return None


# HEADING_LABELS intentionally narrower than converter.py:302–307 —
# converter.py includes PAGE_HEADER / PAGE_FOOTER so running headers/
# footers appear in chunk-level section_path strings. For graph-visible
# SECTION nodes those crumbs are noise (cosmetic repeats per page), so
# the walker excludes them. Deliberate divergence per spec §3.3.
_HEADING_LABELS = (
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.TITLE,
)


_CAPTION_PREFIX_RE = re.compile(
    r"^(figure|fig\.?|table|tbl\.?)\s",
    re.IGNORECASE,
)
_CAPTION_LABEL_RE = re.compile(
    r"^(figure|fig\.?|table|tbl\.?)\s+[\w.-]+",
    re.IGNORECASE,
)


def _caption_label(item) -> str | None:
    """Best-effort short caption label for a picture/table item.

    Scans the item's captions list for the first text starting with a
    "Figure"/"Fig."/"Table"/"Tbl." prefix and returns the prefix-plus-
    identifier substring (e.g., "Figure 3-12" from a longer caption).
    Returns None if no prefixed caption exists. Identity does not depend
    on this — it uses ``self_ref``.
    """
    captions = getattr(item, "captions", None) or []
    for cap in captions:
        # Captions may be RefItem (with $ref) or TextItem (with text).
        text = getattr(cap, "text", None)
        if not text:
            continue
        text = text.strip()
        if _CAPTION_PREFIX_RE.match(text):
            m = _CAPTION_LABEL_RE.match(text)
            if m:
                return m.group(0)
            return text.split(".")[0].strip()
    return None


def _build_section_path_string(path_tuple: tuple[str, ...]) -> str | None:
    """Build the breadcrumb `"A > B > C"` string TextChunk uses (spec §3.4)."""
    if not path_tuple:
        return None
    return " > ".join(path_tuple)


def walk(
    docling_doc_json: dict,
    document_uuid: str,
    pipeline_run_id: str,
    ontology: dict,
) -> MergedExtraction:
    """Emit ontology DOCUMENT / SECTION / FIGURE / TABLE entities + edges
    derived from a DoclingDocument. Spec §3.3.

    Args:
      docling_doc_json: the persisted docling_document.json dict (from
          ``_build_docling_document_json`` at app/workers/pipeline.py:923).
      document_uuid: internal document UUID; used for
          ``identity_scope='document'`` scoping and stamped as
          ``document_id`` property on SECTION/FIGURE/TABLE vertices.
      pipeline_run_id: propagated into ``MergedExtraction.pipeline_run_id``.
      ontology: active ontology dict; passed through to component-identity
          branches inside ``_build_logical_identity``.
    Returns:
      MergedExtraction with ``entities`` (DOCUMENT? + N SECTION + M FIGURE
      + K TABLE) and ``edges`` (HAS_* when DOCUMENT emitted, plus CHILD_OF
      for hierarchical SECTION nesting).
    """
    docling_doc = DoclingDocument.model_validate(docling_doc_json)

    # --- Conditional ontology DOCUMENT -------------------------------------
    document_number = _extract_document_number_from_front_matter(docling_doc)
    doc_entity: DocumentEntity | None = (
        DocumentEntity(document_number=document_number)
        if document_number is not None
        else None
    )

    # --- SECTION enumeration via section_stack mirror ----------------------
    section_stack: list[tuple[int, str]] = []
    section_by_path: "OrderedDict[tuple[str, ...], SectionEntity]" = OrderedDict()
    sibling_counters: dict[tuple[str, ...], int] = defaultdict(int)

    def _register_section(path_tuple: tuple[str, ...]) -> None:
        if not path_tuple or path_tuple in section_by_path:
            return
        parent_tuple = path_tuple[:-1]
        sibling_counters[parent_tuple] += 1
        idx = sibling_counters[parent_tuple]
        parent_section = section_by_path.get(parent_tuple)
        parent_number = parent_section.section_number if parent_section else None
        section_number = f"{parent_number}.{idx}" if parent_number else str(idx)
        section_by_path[path_tuple] = SectionEntity(
            section_number=section_number,
            heading=path_tuple[-1],
            section_path=_build_section_path_string(path_tuple),
        )

    def _ensure_root_section() -> None:
        """Insert synthetic section_number='0' SectionEntity at
        section_by_path[()] if not already present. Idempotent. Single
        code path for both 'zero-headings doc' and 'picture before first
        heading' cases (design §4.1a)."""
        if () not in section_by_path:
            section_by_path[()] = SectionEntity(
                section_number="0",
                heading=None,
                section_path=None,
            )

    # §4.1 — per-picture/table section-stack + reading-order capture.
    pic_to_section:     dict[str, tuple[str, ...]] = {}
    tbl_to_section:     dict[str, tuple[str, ...]] = {}
    pic_to_order_index: dict[str, int] = {}
    tbl_to_order_index: dict[str, int] = {}
    all_items_in_order: list = []

    for order_index, (item, tree_depth) in enumerate(docling_doc.iterate_items()):
        all_items_in_order.append(item)
        label = getattr(item, "label", None)
        text = getattr(item, "text", None) or ""
        if label in _HEADING_LABELS and text.strip():
            # SectionHeaderItem carries a declared `level` attribute that
            # reflects the heading's own semantic level (H1/H2/H3). In
            # well-formed Docling trees that matches iterate_items'
            # tree_depth, but for synthetic docs built via add_heading the
            # tree is flat (tree_depth=1 for every heading) while the
            # declared level is the authoritative hierarchy signal. Prefer
            # the declared level; fall back to tree_depth; default to 1.
            item_level = getattr(item, "level", None)
            if item_level and item_level > 0:
                heading_level = item_level
            elif tree_depth and tree_depth > 0:
                heading_level = tree_depth
            else:
                heading_level = 1
            if label == DocItemLabel.TITLE:
                heading_level = 1
            while section_stack and section_stack[-1][0] >= heading_level:
                section_stack.pop()
            section_stack.append((heading_level, text.strip()))
        path_tuple = tuple(entry[1] for entry in section_stack)
        _register_section(path_tuple)

        if isinstance(item, (PictureItem, TableItem)):
            if not section_stack:
                _ensure_root_section()
            key = tuple(entry[1] for entry in section_stack)
            if isinstance(item, PictureItem):
                pic_to_section[item.self_ref] = key
                pic_to_order_index[item.self_ref] = order_index
            else:
                tbl_to_section[item.self_ref] = key
                tbl_to_order_index[item.self_ref] = order_index

    # End-of-traversal fallback — covers zero-headings AND zero-anchored-content.
    if not section_by_path:
        _ensure_root_section()

    # --- FIGURE + TABLE entities -------------------------------------------
    figures: list[FigureEntity] = []
    for pic in docling_doc.pictures:
        figures.append(
            FigureEntity(
                figure_ref=pic.self_ref,
                figure_label=_caption_label(pic),
            )
        )
    tables: list[TableEntity] = []
    for tbl in docling_doc.tables:
        tables.append(
            TableEntity(
                table_ref=tbl.self_ref,
                table_label=_caption_label(tbl),
            )
        )

    # --- Edges -------------------------------------------------------------
    edges: list[MergedEdgeRecord] = []

    def _identity(entity):
        cfg = getattr(entity, "model_config", {}) or {}
        entity_type = cfg["ontology_name"]
        # Walker-sourced entities use model_config graph_id_fields directly;
        # _build_logical_identity handles components (is_entity=False) via
        # the A0-1 content-based branch.
        if cfg.get("is_entity") is False:
            return _build_logical_identity(entity_type, entity, ontology, document_uuid)
        id_fields = tuple(cfg.get("graph_id_fields") or ())
        scope = cfg.get("identity_scope", "document")
        identity_tuple = tuple(getattr(entity, f, None) for f in id_fields)
        from app.services.extraction_merge import LogicalIdentity
        return LogicalIdentity(
            entity_type=entity_type,
            identity_field_names=id_fields,
            identity_tuple=identity_tuple,
            scope=scope,
            document_id=document_uuid if scope == "document" else None,
        )

    if doc_entity is not None:
        doc_identity = _identity(doc_entity)
        for section in section_by_path.values():
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity,
                to_identity=_identity(section),
                rel_type="HAS_SECTION",
                confidence=1.0,
                pass_origins={"document_anchors"},
            ))
        for fig in figures:
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity,
                to_identity=_identity(fig),
                rel_type="HAS_FIGURE",
                confidence=1.0,
                pass_origins={"document_anchors"},
            ))
        for tbl in tables:
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity,
                to_identity=_identity(tbl),
                rel_type="HAS_TABLE",
                confidence=1.0,
                pass_origins={"document_anchors"},
            ))

    # CHILD_OF edges — hierarchical, independent of ontology DOCUMENT.
    for path_tuple, section in section_by_path.items():
        parent_tuple = path_tuple[:-1]
        parent_section = section_by_path.get(parent_tuple)
        if parent_section is None:
            continue
        edges.append(MergedEdgeRecord(
            from_identity=_identity(section),
            to_identity=_identity(parent_section),
            rel_type="CHILD_OF",
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))

    # --- MergedEntityRecord construction -----------------------------------
    entity_models: list = []
    if doc_entity is not None:
        entity_models.append(doc_entity)
    entity_models.extend(section_by_path.values())
    entity_models.extend(figures)
    entity_models.extend(tables)
    merged_entities = [
        _to_merged_entity_record(m, ontology, document_uuid)
        for m in entity_models
    ]

    return MergedExtraction(
        entities=merged_entities,
        edges=edges,
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id=pipeline_run_id,
        document_id=document_uuid,
    )
