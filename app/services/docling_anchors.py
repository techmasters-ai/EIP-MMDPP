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

from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableItem, TextItem

from app.services.extraction_merge import (
    MergedEdgeRecord,
    MergedExtraction,
    _build_logical_identity,
    _to_merged_entity_record,
)
from ontology_bundles.air_defense_v3.entities import (
    DocumentEntity,
    FigureEntity,
    ImageEntity,
    SectionEntity,
    TableEntity,
    TextBlockEntity,
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


def _resolve_caption_text(cap, docling_doc) -> str | None:
    """Return the caption text string from either an inline TextItem
    (has ``.text``) or a RefItem (has ``.cref`` pointing into the doc body).

    RefItem resolution walks the ``#/section/index`` path on the validated
    DoclingDocument. Defensive — returns None on any resolution failure.
    """
    text = getattr(cap, "text", None)
    if text:
        return text
    cref = getattr(cap, "cref", None)
    if not cref or docling_doc is None:
        return None
    try:
        path = cref.removeprefix("#/") if cref.startswith("#/") else cref
        parts = path.split("/")
        obj = docling_doc
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return getattr(obj, "text", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _caption_label(item, docling_doc=None) -> str | None:
    """Best-effort short caption label for a picture/table item.

    Scans the item's captions list for the first text starting with a
    "Figure"/"Fig."/"Table"/"Tbl." prefix and returns the prefix-plus-
    identifier substring (e.g., "Figure 3-12" from a longer caption).
    Returns None if no prefixed caption exists. Identity does not depend
    on this — it uses ``self_ref``.

    ``docling_doc`` (a validated DoclingDocument) is used to resolve
    RefItem captions (``cref`` pointers) which are the standard form after
    a model_dump / model_validate round-trip. Pass None to skip resolution
    (backward-compat for callers that only have the item).
    """
    captions = getattr(item, "captions", None) or []
    for cap in captions:
        # Captions are RefItem (cref pointer) after round-trip, or
        # TextItem (text attribute) on live objects.
        text = _resolve_caption_text(cap, docling_doc)
        if not text:
            continue
        text = text.strip()
        if _CAPTION_PREFIX_RE.match(text):
            m = _CAPTION_LABEL_RE.match(text)
            if m:
                return m.group(0)
            return text.split(".")[0].strip()
    return None


def _first_prov_page(item) -> int | None:
    """Return prov[0].page_no if available, else None."""
    try:
        prov = getattr(item, "prov", None)
        if prov:
            return prov[0].page_no
    except (AttributeError, IndexError):
        pass
    return None


def _first_prov_bbox(item) -> dict | None:
    """Return prov[0].bbox as a dict {l, t, r, b, page, coord_origin} if
    available, else None. Used for ImageEntity.bbox persistence."""
    try:
        prov = getattr(item, "prov", None)
        if not prov:
            return None
        bbox = prov[0].bbox
        coord_origin = getattr(bbox, "coord_origin", None)
        return {
            "l": bbox.l,
            "t": bbox.t,
            "r": bbox.r,
            "b": bbox.b,
            "page": prov[0].page_no,
            "coord_origin": coord_origin.value if coord_origin is not None else None,
        }
    except (AttributeError, IndexError):
        return None


def _classify_image_role(pic, docling_doc, *, label: str | None) -> str:
    """Heuristic role assignment (design §4.2). Returns HEADER_LOGO,
    INLINE_IMAGE, or UNCAPTIONED_FIGURE.

    ``label`` is the pre-computed _caption_label result (may be None).
    Accepting it as a kwarg avoids re-running caption resolution.

    Geometry lookup: docling_doc.pages[page_no].size.width/height. When
    page info is missing (defensive), skip rule 1 and fall through.
    """
    page_no = None
    try:
        prov = getattr(pic, "prov", None)
        if prov:
            page_no = prov[0].page_no
    except (AttributeError, IndexError):
        pass

    if page_no is not None and page_no in getattr(docling_doc, "pages", {}):
        page = docling_doc.pages[page_no]
        page_width = page.size.width
        page_height = page.size.height
        page_area = page_width * page_height
        try:
            bbox = pic.prov[0].bbox
            if page_no == 1 and bbox.t < page_height / 2:
                pic_area = (bbox.r - bbox.l) * (bbox.b - bbox.t)
                if page_area > 0 and pic_area / page_area < 0.10:
                    return "HEADER_LOGO"
        except (AttributeError, IndexError):
            pass

    # Rule 2 — caption present but no "Figure N" prefix already classifies as UNCAPTIONED_FIGURE.
    if label:
        return "UNCAPTIONED_FIGURE"
    return "INLINE_IMAGE"


def _is_valid_near_text(item, captions_linked: set[str]) -> bool:
    """Accept body-text items only: not section headers, titles, or
    captions already linked to pictures/tables. Design §4.3."""
    if not isinstance(item, TextItem):
        return False
    label = getattr(item, "label", None)
    if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.CAPTION):
        return False
    if getattr(item, "self_ref", None) in captions_linked:
        return False
    return True


def _neighbors(target_order: int, items: list, captions_linked: set[str], window: int = 2) -> list:
    """Return up to ``window`` valid-near-text items before + ``window``
    after target_order, in reading order. Design §4.3."""
    before: list = []
    i = target_order - 1
    while i >= 0 and len(before) < window:
        if _is_valid_near_text(items[i], captions_linked):
            before.append(items[i])
        i -= 1
    after: list = []
    j = target_order + 1
    while j < len(items) and len(after) < window:
        if _is_valid_near_text(items[j], captions_linked):
            after.append(items[j])
        j += 1
    return list(reversed(before)) + after


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

    # --- FIGURE + IMAGE + TABLE entities -------------------------------------------
    figures: list[FigureEntity] = []
    images:  list[ImageEntity]  = []
    pic_entities: list = []   # parallel to docling_doc.pictures for §4.3 zip safety

    for pic in docling_doc.pictures:
        label = _caption_label(pic, docling_doc)
        if label and label.lower().startswith(("figure", "fig")):
            entity = FigureEntity(
                figure_ref=pic.self_ref,
                figure_label=label,
                storage_key=None,
            )
            figures.append(entity)
        else:
            entity = ImageEntity(
                image_ref=pic.self_ref,
                page=_first_prov_page(pic),
                caption=label,  # may be None or a non-"Figure" caption
                storage_key=None,
                bbox=_first_prov_bbox(pic),
                image_role=_classify_image_role(pic, docling_doc, label=label),
            )
            images.append(entity)
        pic_entities.append(entity)

    tables: list[TableEntity] = []
    for tbl in docling_doc.tables:
        tables.append(
            TableEntity(
                table_ref=tbl.self_ref,
                table_label=_caption_label(tbl, docling_doc),
            )
        )

    # §4.3 — captions_linked: self_refs already owned by pictures/tables as captions.
    captions_linked: set[str] = set()
    for item in list(docling_doc.pictures) + list(docling_doc.tables):
        caps = getattr(item, "captions", None) or []
        for cap in caps:
            ref = getattr(cap, "cref", None)
            if isinstance(ref, str):
                captions_linked.add(ref)

    # §4.3 — lazy TEXT_BLOCK emission + NEAR_TEXT edge pairs.
    text_blocks_by_ref: dict[str, TextBlockEntity] = {}
    near_text_pairs: list[tuple] = []  # (parent_entity, text_block_entity)

    # pic_entities is parallel to docling_doc.pictures (§3c) — zip is safe.
    for pic, entity in zip(docling_doc.pictures, pic_entities, strict=True):
        order_index = pic_to_order_index.get(pic.self_ref)
        if order_index is None:
            continue
        for text_item in _neighbors(order_index, all_items_in_order, captions_linked):
            self_ref = getattr(text_item, "self_ref", None)
            if not isinstance(self_ref, str):
                continue
            tb = text_blocks_by_ref.get(self_ref)
            if tb is None:
                text = (getattr(text_item, "text", None) or "")[:500]
                label = getattr(text_item, "label", None)
                label_value = label.value if hasattr(label, "value") else (label if isinstance(label, str) else None)
                prov_page = _first_prov_page(text_item)
                tb = TextBlockEntity(
                    text_ref=self_ref,
                    text=text,
                    label=label_value,
                    page=prov_page,
                )
                text_blocks_by_ref[self_ref] = tb
            near_text_pairs.append((entity, tb))

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

    # §4.3 — NEAR_TEXT edges. pic_entities iteration above populated near_text_pairs.
    for parent_entity, tb in near_text_pairs:
        edges.append(MergedEdgeRecord(
            from_identity=_identity(parent_entity),
            to_identity=_identity(tb),
            rel_type="NEAR_TEXT",
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))

    # --- MergedEntityRecord construction -----------------------------------
    entity_models: list = []
    if doc_entity is not None:
        entity_models.append(doc_entity)
    entity_models.extend(section_by_path.values())
    entity_models.extend(figures)
    entity_models.extend(images)       # uncaptioned pictures as IMAGE entities
    entity_models.extend(tables)
    entity_models.extend(text_blocks_by_ref.values())   # §4.3 — lazy TEXT_BLOCK neighbors
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
