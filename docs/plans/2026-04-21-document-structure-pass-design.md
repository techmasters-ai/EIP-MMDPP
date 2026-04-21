# Document Structure Extraction — Design (v2.3)

**Date:** 2026-04-21
**Branch target:** `feat/pydantic-ontology-ssot` (or follow-on feature branch)
**Bundle:** `air_defense_v3`
**Task IDs:** #1 (relationships + matrix), #2 (schemas), #3 (walker), #5 (tests), #6 (notebooks).

## Revision history

- **v1** — proposed a new LLM-backed pass. Invalidated: `app/services/docling_anchors.py` already does structural extraction.
- **v2** — rescoped to "extend docling_anchors".
- **v2.1** — CHILD_OF enum requirement + split storage_key story.
- **v2.2** — six scope corrections (enum gap wider than CHILD_OF; correct metadata-surface name; lazy-seed for pre-heading anchored content; drop the contradicted `picture_storage_keys` kwarg; mandatory ALL_ENTITIES registration; validation_matrix is ontology hygiene not runtime unblock).
- **v2.3 (this version)** — six more issues from user review:
  1. Storage-key code sketch still references a `_resolve_picture_storage_key(pic)` helper and populates FIGURE/IMAGE `storage_key` — stale. §4.2 now always passes `storage_key=None` on FIGURE and IMAGE. Goal narrowed to DocumentEntity-only.
  2. Deleting the existing zero-heading fallback regresses documents with zero headings AND zero pictures/tables. §4.1a now keeps the fallback as an **explicit end-of-traversal** `_ensure_root_section()` call instead of deleting it.
  3. `zip(docling_doc.pictures, figures + images)` in §4.3 mis-aligns when captioned and uncaptioned pictures are interleaved. Replaced with a parallel `pic_entities` list aligned with `docling_doc.pictures` order.
  4. §4.1 data structures were described as `dict[int, ...]` in prose but keyed by `item.self_ref` in code. Types + prose fixed to `dict[str, ...]`.
  5. HEADER_LOGO heuristic needed a page-geometry source. §4.2 now specifies `docling_doc.pages[page_no].size.width/height`.
  6. Parity tests (`test_relationships_parity.py`, `test_validation_matrix_parity.py`) were *deleted* in commit 78c7d51 (F-4 cleanup). Spec no longer claims to update them. Correct existing test file is `tests/unit/test_docling_anchor_walker.py`; correct fixtures dir is `tests/fixtures/docling_anchors/`; integration smoke test already at `tests/integration/test_docling_anchors_smoke.py`.

## Scope correction (why this isn't a new pass)

The v1 of this design proposed a **new** extraction pass for document structure. An independent spec review caught that this duplicates existing infrastructure: `app/services/docling_anchors.py` already walks the DoclingDocument and emits DOCUMENT / SECTION / FIGURE / TABLE entities deterministically, wired into the Celery pipeline at `app/workers/pipeline.py:1681, 1818, 4261-4316` as the `derive_document_anchors` task. The existing walker's docstring says "Replaces the LLM reference pass (deleted in C-1)" — the B-decision from our conversation was already made and shipped.

This v2 scopes the work to **extend the existing walker**, adding the pieces that don't exist yet: IMAGE entity, TEXT_BLOCK entity, NEAR_TEXT edges, SECTION-level attribution for figures/tables/images, and the missing validation-matrix rows for the anchor-emitted relationships.

## Problem

What the current walker does **well**:
- DOCUMENT conditionally emitted when a MIL-STD / TM designator appears in front matter (`_extract_document_number_from_front_matter`).
- SECTION via auto-numbered synthetic `section_number` ("1", "1.1", "1.1.1") from the `section_stack` traversal.
- FIGURE per `doc.pictures[i]` with `figure_ref=pic.self_ref`, `figure_label` = best-effort caption match.
- TABLE per `doc.tables[i]` with `table_ref=tbl.self_ref`, `table_label` = best-effort caption match.
- HAS_SECTION / HAS_FIGURE / HAS_TABLE edges from DOCUMENT when DOCUMENT exists.
- CHILD_OF edges for hierarchical SECTION nesting (independent of DOCUMENT).

What the current walker **does not do** (gaps this design closes):
1. Distinguishes captioned figures ("Figure 3-12") from uncaptioned embedded pictures (header logos, inline images, decorative photos). All pictures become FIGURE today.
2. Links FIGURE / TABLE to the SECTION that contains them — only to DOCUMENT. There is no section-level attribution.
3. Records any surrounding-text context for figures / tables / images — downstream retrieval has no graph path from an image to the paragraphs that reference it.
4. Populates `storage_key` on DOCUMENT or FIGURE, so downstream graph→MinIO retrieval requires a SQL join.

Additionally, the review surfaced an ontology-consistency gap: **neither the `RelationshipType` enum nor `VALIDATION_MATRIX` currently carries HAS_SECTION / HAS_FIGURE / HAS_TABLE / CHILD_OF**, even though the walker emits them. This is *not* causing runtime drops — `derive_document_anchors` at `pipeline.py:4316-4347` writes anchor edges directly via `create_structural_edge_sync` and does **not** consult the ontology relationship-validation path (pipeline.py:4277-4278 docstring says so explicitly). So the existing system works today because it skips validation, not because the matrix is right. We still add the missing enum members + matrix triples in this change so that ontology introspection, parity tests, query profiles, and any future consumer that *does* read the matrix agree with reality. This is ontology hygiene, not a runtime bug fix.

## Goal

Extend `docling_anchors.py` so that every extracted document has:
- **IMAGE** entities for pictures without a "Figure N"-style caption.
- **TEXT_BLOCK** entities lazily emitted as the ±2 reading-order neighbors around each IMAGE and FIGURE.
- **NEAR_TEXT** edges from IMAGE and FIGURE to their surrounding TEXT_BLOCKs.
- **HAS_IMAGE** edges from DOCUMENT to IMAGE, following the existing HAS_SECTION / HAS_FIGURE / HAS_TABLE pattern.
- **Section-level attribution**: HAS_FIGURE, HAS_TABLE, HAS_IMAGE edges from the enclosing SECTION to the FIGURE / TABLE / IMAGE, in addition to the existing DOCUMENT→* edges.
- **`storage_key`** populated on DOCUMENT only in this change (source file MinIO key from the Document SQL row). FIGURE and IMAGE have the schema field available but always null at emission — per-picture MinIO plumbing needs an `Artifact.self_ref` DB migration that is out of scope here.

Plus: add the missing HAS_SECTION / HAS_FIGURE / HAS_TABLE / CHILD_OF triples to `validation_matrix.py` alongside the new HAS_IMAGE / NEAR_TEXT triples.

Out of scope (follow-on change, not this one):
- Downstream LLM passes (`radar_domain`, `missile_domain`, `system_links`) consuming the new structural refs — e.g. `RADAR_SYSTEM --MENTIONED_IN--> SECTION`.
- Graph-viz / UI surface changes — the viewer picks up new vertex/edge types automatically.
- Migration of already-ingested documents — see "Migration" section below.

## 1. Architecture — no new endpoint, no new manifest pass

**Everything runs inside the existing `derive_document_anchors` Celery task.** No new endpoint on docling-graph, no new manifest pass, no worker routing changes. The carrier is `app/services/docling_anchors.py::walk()`, which is called once per document at pipeline.py:4316 and returns a `MergedExtraction` that the merger consumes.

We extend `walk()` with additional emission branches and new edge records. The signature gains **one** new keyword argument: `source_storage_key: str | None = None` (see §4.4). No per-picture storage plumbing in this change.

**Why not a new endpoint / pass**: the structural extraction is already fully deterministic, already wired, already runs on every document. Adding a parallel path would duplicate work and race the existing one. The review was right; v1 was wrong.

## 2. Entities and schemas

Four entities exist in `ontology_bundles/air_defense_v3/entities.py` today (`DocumentEntity`, `SectionEntity`, `FigureEntity`, `TableEntity`). Additions:

### New — `ImageEntity`

```python
class ImageEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="IMAGE",
        graph_id_fields=["image_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )
    image_ref: str                        # identity — self_ref, e.g. "#/pictures/12"
    document_id: Optional[str] = None
    page: Optional[int] = None
    caption: Optional[str] = None
    mime_type: Optional[str] = None
    storage_key: Optional[str] = None
    bbox: Optional[dict] = None           # {l, t, r, b, page, coord_origin}
    image_role: Optional[str] = None      # HEADER_LOGO | INLINE_IMAGE | UNCAPTIONED_FIGURE
    confidence: Optional[float] = 1.0
```

Declared before TextBlockEntity or after — either works, since docling_anchors builds MergedEdgeRecord instances directly (not via Pydantic typed-edge fields), so no forward reference is needed on the model.

### New — `TextBlockEntity`

```python
class TextBlockEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="TEXT_BLOCK",
        graph_id_fields=["text_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )
    text_ref: str                         # identity — self_ref, e.g. "#/texts/237"
    document_id: Optional[str] = None
    text: Optional[str] = None            # 500-char truncated preview
    label: Optional[str] = None           # TEXT | PARAGRAPH | LIST_ITEM
    page: Optional[int] = None
    confidence: Optional[float] = 1.0
```

### Additions to existing entities

- `DocumentEntity` gains `storage_key: Optional[str] = None`.
- `FigureEntity` gains `storage_key: Optional[str] = None`.

### Identity-collision note

Review flagged that the SECTION fallback path (`section_number="#/texts/237"`) could look like a TEXT_BLOCK ref. In this walker, SECTION `section_number` is **synthetic** (auto-incrementing "1", "1.1"), not heading-derived, so the collision case doesn't arise. Even if it did, `LogicalIdentity` is keyed on `entity_type` first — SECTION and TEXT_BLOCK entities with equal string identities would still be distinct graph vertices. No change needed.

### ALL_ENTITIES registry update (required, not optional)

`ontology_bundles/air_defense_v3/entities.py:1285-1330` has an `ALL_ENTITIES: dict[str, type[BaseModel]]` registry keyed by `ontology_name`. Lines 1333-1334 then loop `for _cls in ALL_ENTITIES.values(): _cls.model_rebuild()` to resolve Pydantic forward references. Introspection consumers (ontology-introspect scripts, the bundle loader, schema-emission code) all read this registry.

**Required additions** (unconditional — the spec earlier hedged with "if present"; the file makes it unambiguous):

```python
ALL_ENTITIES: dict[str, type[BaseModel]] = {
    ...
    "IMAGE": ImageEntity,
    "TEXT_BLOCK": TextBlockEntity,
    ...
}
```

Both new entity classes must appear there, or `model_rebuild()` won't run for them and any parity / introspection test that enumerates the registry will miss them.

### What does NOT change

- `ontology_bundles/air_defense_v3/extraction_schemas/` — no new file. This work is not a new LLM pass.
- `manifest.yaml` — unchanged. No new pass entry, no `extraction_method` flag.
- The existing `SectionEntity`, `FigureEntity`, `TableEntity`, `DocumentEntity` shapes — apart from the new `storage_key` field on DocumentEntity + FigureEntity (§2).

## 3. Relationships and validation matrix

### Additions to `ontology_bundles/air_defense_v3/relationships.py`

Add **six** members to `RelationshipType` (`ontology_bundles/air_defense_v3/relationships.py:20-76`). The v2.1 spec under-counted: the walker emits HAS_SECTION, HAS_FIGURE, HAS_TABLE, and CHILD_OF as raw strings today, and *none* of the four is in the enum. Adding the new IMAGE / TEXT_BLOCK relations without also adding the anchor relations means the `RelationshipType.HAS_SECTION` (etc.) references in the new VALIDATION_MATRIX triples would `AttributeError` at import.

```python
HAS_SECTION = "HAS_SECTION"   # missing today — walker emits as raw string
HAS_FIGURE  = "HAS_FIGURE"    # missing today — walker emits as raw string
HAS_TABLE   = "HAS_TABLE"     # missing today — walker emits as raw string
CHILD_OF    = "CHILD_OF"      # missing today — walker emits as raw string at docling_anchors.py:295
HAS_IMAGE   = "HAS_IMAGE"     # new in this change
NEAR_TEXT   = "NEAR_TEXT"     # new in this change
```

Add matching descriptor dicts to **`_STATIC_RELATIONSHIP_METADATA`** at `relationships.py:128` (not "RELATIONSHIPS list" — that name was wrong in v2.1). The metadata surfaces built from `_STATIC_RELATIONSHIP_METADATA` are `RelationshipMetadata` (class, line 82) and `RELATIONSHIP_METADATA` (dict, line 208, derived from the static list at build time). Adding descriptor dicts to `_STATIC_RELATIONSHIP_METADATA` automatically propagates to `RELATIONSHIP_METADATA`.

```python
{"name": "HAS_SECTION","label": "Has Section","description": "Document or parent contains a section", "source_type": None, "target_type": "SECTION", "cardinality": "one_to_many"},
{"name": "HAS_FIGURE", "label": "Has Figure", "description": "Document or section contains a captioned figure", "source_type": None, "target_type": "FIGURE", "cardinality": "one_to_many"},
{"name": "HAS_TABLE",  "label": "Has Table",  "description": "Document or section contains a table", "source_type": None, "target_type": "TABLE", "cardinality": "one_to_many"},
{"name": "CHILD_OF",   "label": "Child Of",   "description": "Hierarchical containment (e.g. sub-section within parent section)", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
{"name": "HAS_IMAGE",  "label": "Has Image",  "description": "Document or section contains an uncaptioned image or embedded picture", "source_type": None, "target_type": "IMAGE", "cardinality": "one_to_many"},
{"name": "NEAR_TEXT",  "label": "Near Text",  "description": "Figure or image appears near a text block in reading order", "source_type": None, "target_type": "TEXT_BLOCK", "cardinality": "one_to_many"},
```

### Additions to `ontology_bundles/air_defense_v3/validation_matrix.py`

Adds to `VALIDATION_MATRIX`:

```python
# Structural anchors — the existing walker emits these relationship
# types today but they were never added to the matrix. Lockstep fix.
("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),

# Section-level attribution — new in this change.
("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),

# IMAGE — new entity type.
("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),

# Near-text context — new in this change.
("FIGURE",   RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
("IMAGE",    RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
```

No parity-test update needed: `tests/unit/test_relationships_parity.py` and `tests/unit/test_validation_matrix_parity.py` were deleted in commit `78c7d51` (F-4 cleanup) along with their frozen ontology.yaml snapshot. The Python ontology files are the sole source of truth today.

## 4. Walker changes — `docling_anchors.py`

All walker changes live in `app/services/docling_anchors.py::walk()`. The existing function's flow stays intact; we add new branches.

### 4.1 Capture the section stack at picture/table emission time

Currently, pictures and tables are emitted in two passes **after** the `iterate_items()` loop at line 184 has finished, so there is no surviving link between a picture/table and its enclosing section at emission time.

**Change:** during `iterate_items()`, track the current section_stack alongside each picture/table encountered. Build four dicts keyed by **self_ref** (string):

```python
pic_to_section:      dict[str, tuple[str, ...]] = {}
tbl_to_section:      dict[str, tuple[str, ...]] = {}
pic_to_order_index:  dict[str, int] = {}
tbl_to_order_index:  dict[str, int] = {}

for order_index, (item, tree_depth) in enumerate(docling_doc.iterate_items()):
    # ... existing section_stack logic ...
    if isinstance(item, PictureItem):
        pic_to_section[item.self_ref] = tuple(e[1] for e in section_stack)
        pic_to_order_index[item.self_ref] = order_index
    elif isinstance(item, TableItem):
        tbl_to_section[item.self_ref] = tuple(e[1] for e in section_stack)
        tbl_to_order_index[item.self_ref] = order_index
```

Lookup `section_by_path[pic_to_section[pic.self_ref]]` to get the enclosing `SectionEntity`. Empty tuple (picture appeared before any heading) is handled by §4.1a's lazy-seed.

### 4.1a Lazy-seed synthetic root section for pre-heading anchored content

The existing fallback at `docling_anchors.py:210-216` only creates the synthetic `section_number="0"` SectionEntity when the document has *zero* headings. In a document that has headings later, a picture/table appearing before the first heading leaves `path_tuple = ()` and `section_by_path.get(())` returns `None` — the proposed SECTION→FIGURE/TABLE/IMAGE edges would be silently dropped.

Fix: during the section-stack tracking in §4.1, when a picture or table is encountered with an empty `section_stack`, lazily seed the synthetic root SectionEntity once:

```python
def _ensure_root_section() -> None:
    """Seed the synthetic section_number='0' SectionEntity if we see
    anchored content before any real heading. Idempotent."""
    if () not in section_by_path:
        section_by_path[()] = SectionEntity(
            section_number="0",
            heading=None,
            section_path=None,
        )

for order_index, (item, tree_depth) in enumerate(docling_doc.iterate_items()):
    # ... existing section_stack logic ...
    if isinstance(item, (PictureItem, TableItem)) and not section_stack:
        _ensure_root_section()
    # ... record pic_to_section / tbl_to_section as before ...
```

**Do not delete the existing zero-headings fallback** at lines 210-216 outright — that would regress headingless documents with zero pictures/tables (no iteration branch would fire `_ensure_root_section`). Instead, rewrite that fallback as an explicit call after the traversal finishes:

```python
# End of iterate_items() loop. Covers: zero headings AND zero anchored
# content (documents of pure body text) → still get one SectionEntity.
if not section_by_path:
    _ensure_root_section()
```

That's equivalent in behavior to the current fallback, but uses the same helper so there's only one code path for the synthetic root section.

### 4.2 Split pictures into FIGURE vs IMAGE

Replace the loop at lines 220-226:

```python
figures: list[FigureEntity] = []
images:  list[ImageEntity]  = []
pic_entities: list[Any] = []     # parallel to docling_doc.pictures — used in §4.3 for NEAR_TEXT

for pic in docling_doc.pictures:
    label = _caption_label(pic)   # returns "Figure 3-12" or None
    # storage_key always None on picture-derived entities this change;
    # per-picture plumbing deferred pending Artifact.self_ref migration (§4.4).
    if label and label.lower().startswith(("figure", "fig")):
        entity = FigureEntity(
            figure_ref=pic.self_ref,
            figure_label=label,
            page=_first_prov_page(pic),
            caption=_full_caption(pic),
            storage_key=None,
        )
        figures.append(entity)
    else:
        entity = ImageEntity(
            image_ref=pic.self_ref,
            page=_first_prov_page(pic),
            caption=_full_caption(pic),
            storage_key=None,
            bbox=_first_prov_bbox(pic),
            image_role=_classify_image_role(pic, docling_doc),
        )
        images.append(entity)
    pic_entities.append(entity)
```

`_classify_image_role(pic, docling_doc)` heuristics (page geometry comes from `docling_doc.pages[page_no].size`, which is `docling_core.types.doc.Size(width, height)` in pixels or points depending on the source; Docling normalizes this to the page's native coordinate system):
- `page_no = _first_prov_page(pic); bbox = _first_prov_bbox(pic)`
- Compute `page_area = pages[page_no].size.width * pages[page_no].size.height` if the page exists in `docling_doc.pages`.
- Rule 1 — `page_no == 1` AND `bbox.t < page_area.height / 2` AND `(bbox.r - bbox.l) * (bbox.b - bbox.t) / page_area < 0.10` → `HEADER_LOGO`.
- Rule 2 — has any caption text but caption doesn't start with "Figure" / "Fig" → `UNCAPTIONED_FIGURE`.
- Rule 3 — otherwise → `INLINE_IMAGE`.

When `page_no` is None or absent from `docling_doc.pages` (defensive — shouldn't happen on well-formed Docling output), skip Rule 1 and fall through to Rule 2 or 3. Keeps the classifier deterministic without requiring page geometry.

### 4.3 Lazy TEXT_BLOCK emission + NEAR_TEXT edges

After figures + images are built, build the reading-order text neighbor list. The walker has already iterated all items; cache the ordered `(order_index, item)` list from the `iterate_items()` pass.

Helper:

```python
def _neighbors(target_order: int, items: list, window: int = 2) -> list:
    """Return up to 2 text items before + 2 after the target in reading order.
    Skips: SECTION_HEADER, captions attached to pictures/tables, other
    pictures/tables, figures in the already-captured caption set."""
    before, after = [], []
    i = target_order - 1
    while i >= 0 and len(before) < window:
        if _is_valid_near_text(items[i]):
            before.append(items[i])
        i -= 1
    j = target_order + 1
    while j < len(items) and len(after) < window:
        if _is_valid_near_text(items[j]):
            after.append(items[j])
        j += 1
    return list(reversed(before)) + after
```

For each emitted figure and image, build TextBlockEntity instances from its neighbors:

```python
text_blocks_by_ref: dict[str, TextBlockEntity] = {}   # dedup
near_text_edges: list[tuple[Any, TextBlockEntity]] = []

def _attach_neighbors(parent_entity, picture_or_table_item):
    for text_item in _neighbors(order_index_of(picture_or_table_item), items):
        tb = text_blocks_by_ref.get(text_item.self_ref)
        if tb is None:
            tb = TextBlockEntity(
                text_ref=text_item.self_ref,
                text=(text_item.text or "")[:500],
                label=getattr(text_item.label, "value", None),
                page=_first_prov_page(text_item),
            )
            text_blocks_by_ref[text_item.self_ref] = tb
        near_text_edges.append((parent_entity, tb))

# pic_entities is built 1:1 with docling_doc.pictures in §4.2, so zip is safe
# here — captioned FIGURE and uncaptioned IMAGE entities stay in document order.
# Do NOT use `figures + images` — that reorders entities relative to document
# order and landed neighbors on the wrong picture when caption types interleaved.
for pic, entity in zip(docling_doc.pictures, pic_entities, strict=True):
    _attach_neighbors(entity, pic)
```

Same is **not** applied to tables in this change — scope creep avoided. TABLE near-text can be a follow-on.

### 4.4 `storage_key` — DocumentEntity only in this change

**DocumentEntity.storage_key** — trivial. The `Document` SQL row already has a `storage_key` column (`app/models/ingest.py:58-59`) for the source file in MinIO. Add one kwarg to `walk()`:

```python
def walk(
    docling_doc_json: dict,
    document_uuid: str,
    pipeline_run_id: str,
    ontology: dict,
    *,
    source_storage_key: str | None = None,   # NEW — populates DocumentEntity.storage_key
) -> MergedExtraction:
```

Call-site at `pipeline.py:4316` reads the Document row (already in scope from `derive_document_anchors`) and passes `source_storage_key=document.storage_key`.

**FigureEntity.storage_key / ImageEntity.storage_key — fields defined, emission deferred.** Per-picture MinIO keys live on the `Artifact` table (`app/models/ingest.py:117`), keyed by `document_id + bounding_box + page_number` — no `self_ref` column. Building a `self_ref → MinIO key` map requires either a DB migration (add `Artifact.self_ref`, populate at artifact-creation) or a fragile bbox fuzzy match. That's scope creep for this change.

**Decision:** FigureEntity.storage_key + ImageEntity.storage_key fields stay on the models (schema is forward-compatible) but are always populated `None` in this change. No `picture_storage_keys` kwarg on `walk()`, no asset-map plumbing, no tests for picture storage resolution. Follow-on change adds `Artifact.self_ref` + the kwarg + its tests together.

### 4.5 Edge additions

In the existing edge-building block (lines 237-298), extend:

**DOCUMENT→IMAGE HAS_IMAGE** (when DOCUMENT is emitted):

```python
if doc_entity is not None:
    for img in images:
        edges.append(MergedEdgeRecord(
            from_identity=doc_identity,
            to_identity=_identity(img),
            rel_type="HAS_IMAGE",
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))
```

**SECTION→FIGURE / SECTION→TABLE / SECTION→IMAGE** (unconditional on DOCUMENT):

```python
for pic in docling_doc.pictures:
    section = section_by_path.get(pic_to_section.get(pic.self_ref, ()))
    if section is None:
        continue
    entity = _entity_for_pic(pic)  # figure or image, already built above
    rel = "HAS_FIGURE" if isinstance(entity, FigureEntity) else "HAS_IMAGE"
    edges.append(MergedEdgeRecord(
        from_identity=_identity(section),
        to_identity=_identity(entity),
        rel_type=rel,
        confidence=1.0,
        pass_origins={"document_anchors"},
    ))
# ... same loop for tables with HAS_TABLE ...
```

**FIGURE/IMAGE →TEXT_BLOCK NEAR_TEXT**:

```python
for parent, tb in near_text_edges:
    edges.append(MergedEdgeRecord(
        from_identity=_identity(parent),
        to_identity=_identity(tb),
        rel_type="NEAR_TEXT",
        confidence=1.0,
        pass_origins={"document_anchors"},
    ))
```

### 4.6 MergedEntityRecord extension

Extend the entity assembly at lines 301-310 to include the new types:

```python
entity_models.extend(images)
entity_models.extend(text_blocks_by_ref.values())
```

## 5. Error handling

Walker is deterministic. Error surface is limited to parser edge cases.

| Case | Behavior |
|---|---|
| Zero pictures / zero tables | Walker emits DOCUMENT (if designator found) + SECTIONs. No FIGUREs, IMAGEs, or TEXT_BLOCKs. Unchanged existing behavior, extended here. |
| Picture with no `self_ref` | Log WARNING, skip. Shouldn't happen in Docling output; defensive. |
| Picture's reading-order neighbors are all section headers / other pictures | Zero TextBlockEntity children for that picture. NEAR_TEXT edges simply not emitted. |
| `storage_keys` kwarg not provided | `storage_key` null on every entity. |
| Picture emitted BEFORE any section header (document has headings later) | **Lazy-seed a synthetic section_number="0"** on first pre-heading anchored-content encounter. Existing fallback at `docling_anchors.py:210-216` only fires when the document has *zero* headings total — not when the first heading appears later. Without lazy seeding, `section_by_path.get(())` returns None and the proposed SECTION→* attribution silently drops the edge. See §4.2a for the code change. |
| Pydantic validation of any emitted model fails | The walker raises — caught by the Celery task wrapper at `pipeline.py:4316` and routed to FAILED stage_run. Indicates a walker bug, not a runtime condition. |

No new diagnostic keys — `MergedExtraction` has no diagnostics dict; the Celery task writes execution stats via `stage_run`. If walker instrumentation becomes valuable later, that's a separate concern.

## 6. Migration

Changing pictures' bucket from FIGURE to IMAGE (when uncaptioned) is a **data migration** for any document already in the graph:

- For every existing FIGURE vertex whose `figure_ref` is a self_ref AND no `figure_label`: delete FIGURE, insert IMAGE with `image_ref=figure_ref`, re-wire incoming HAS_FIGURE → HAS_IMAGE.
- Captioned FIGUREs (have a `figure_label`) stay as FIGURE.

Two paths:

1. **Re-ingest** all documents. Cleanest. Acceptable if prod doesn't have many docs yet. Resets all identities consistently.
2. **One-shot migration script** in `scripts/migrate_uncaptioned_figures_to_images.py` that runs once against ArcadeDB. More code, avoids re-parsing PDFs.

Recommendation: **(1) re-ingest** if the doc count is small (confirm with operator). Otherwise (2).

The design **does not** require picking one now — implementation can land with (1) noted in VERIFICATION_CHECKLIST.md; (2) written later if needed.

## 7. Testing

Extend the existing walker tests at `tests/unit/test_docling_anchor_walker.py` (singular "anchor"; verified present). The fixtures directory `tests/fixtures/docling_anchors/` already holds four reusable DoclingDocument JSONs — `empty_structure.json`, `sa2_minimal.json`, `with_document_number.json`, `with_figures_tables.json`. Add new fixtures only where an existing one doesn't cover the case (e.g. a doc with interleaved captioned/uncaptioned pictures for §4.3 regression, and a doc with a pre-heading picture for §4.1a).

**Note on parity tests:** the prior revisions referenced `tests/unit/test_relationships_parity.py` and `tests/unit/test_validation_matrix_parity.py`. Those were **deleted** in commit `78c7d51 (test(cleanup): delete 2 relationship-parity tests (F-4))` alongside the ontology.yaml snapshot they compared against. No parity layer to update; the Python ontology files are the sole source of truth now. Stale `.pyc` entries in `__pycache__` are incidental.

Additions:

### Unit tests

- `test_uncaptioned_picture_becomes_image` — picture with no caption → ImageEntity, `image_ref=self_ref`, `image_role=INLINE_IMAGE`.
- `test_captioned_figure_still_figure` — picture with caption "Figure 3-12" → FigureEntity, `figure_ref=self_ref`, `figure_label="Figure 3-12"`. Regression.
- `test_header_logo_heuristic` — page-1 small-bbox-top picture → ImageEntity, `image_role=HEADER_LOGO`.
- `test_section_attribution_figure` — a figure emitted inside section "2.1" gets a SECTION→FIGURE HAS_FIGURE edge.
- `test_section_attribution_image` — same for image + HAS_IMAGE.
- `test_section_attribution_table` — same for table + HAS_TABLE.
- `test_pre_heading_picture_falls_back_to_section_zero` — picture before any heading → attributed to synthetic section_number="0".
- `test_near_text_window_figure` — a FigureEntity gets up to 4 TextBlockEntity NEAR_TEXT neighbors (reading-order ±2), skipping section headers.
- `test_near_text_window_image` — same for ImageEntity.
- `test_near_text_dedup` — same text block neighbor to two pictures → one TextBlockEntity, two NEAR_TEXT edges.
- `test_near_text_no_valid_neighbors` — figure surrounded only by other figures / headers → zero NEAR_TEXT edges, no crash.
- `test_document_storage_key_from_kwarg` — when `source_storage_key` kwarg passed, DocumentEntity.storage_key populated; when absent, null. (FIGURE/IMAGE.storage_key always null in this change — no picture-level plumbing.)

### Integration test

Extend the existing `tests/integration/test_docling_anchors_smoke.py` (verified present) rather than introducing a parallel file. Add assertions against a committed DoclingDocument JSON that carries at least one captioned figure, one uncaptioned picture, and one table:

- ≥1 DOCUMENT, ≥3 SECTIONs, ≥1 captioned FIGURE, ≥1 uncaptioned IMAGE, ≥1 TABLE, ≥3 TEXT_BLOCKs
- SECTION→FIGURE HAS_FIGURE, SECTION→IMAGE HAS_IMAGE, SECTION→TABLE HAS_TABLE edges present
- FIGURE/IMAGE → TEXT_BLOCK NEAR_TEXT edges present
- Pre-heading picture attached to the synthetic root SECTION (§4.1a regression check)

Fixture: use `tests/fixtures/docling_anchors/with_figures_tables.json` if it covers these cases; otherwise add a new fixture file alongside the existing four.

### End-to-end

- Existing E2E ingest tests must still pass (they exercise the `derive_document_anchors` task). Update one to assert IMAGE vertices land in ArcadeDB for a doc with uncaptioned pictures.

### Post-implementation

Per memory `feedback_post_code_workflow.md`:
1. Run simplify on all new code.
2. Full test suite.
3. Update `VERIFICATION_CHECKLIST.md` with: (a) new IMAGE + TEXT_BLOCK entity types listed; (b) new HAS_IMAGE + NEAR_TEXT relationships; (c) re-ingestion required for legacy documents if they had uncaptioned pictures.
4. Update README if a public pass list is documented.

## 8. File-level change summary

| File | Change |
|---|---|
| `ontology_bundles/air_defense_v3/entities.py` | Add `ImageEntity`, `TextBlockEntity`. Register both in `ALL_ENTITIES` at line 1285-1330 (required — not optional; `model_rebuild()` + introspection loops depend on it). Add `storage_key: Optional[str] = None` to `DocumentEntity` + `FigureEntity`. |
| `ontology_bundles/air_defense_v3/relationships.py` | Add **six** enum members (HAS_SECTION, HAS_FIGURE, HAS_TABLE, CHILD_OF, HAS_IMAGE, NEAR_TEXT) and matching descriptors to `_STATIC_RELATIONSHIP_METADATA` at line 128. `RELATIONSHIP_METADATA` (line 208) is derived automatically. |
| `ontology_bundles/air_defense_v3/validation_matrix.py` | Add 10 triples (see §3). Not runtime-load-bearing for anchor edges today (they go direct via `create_structural_edge_sync`), but required for ontology consistency + parity tests + any query-profile consumer. |
| `app/services/docling_anchors.py` | §4 walker extensions: lazy-seed synthetic root section when pre-heading anchored content appears; split FIGURE vs IMAGE; attribute pictures/tables to enclosing section; lazily emit TEXT_BLOCK neighbors; NEAR_TEXT edges; `source_storage_key` kwarg populates DocumentEntity.storage_key. No `picture_storage_keys` kwarg — deferred. |
| `app/workers/pipeline.py:4316` | Pass `source_storage_key=document.storage_key` from the Document SQL row. |
| `tests/unit/test_docling_anchor_walker.py` | Extend with §7 unit tests for IMAGE / TEXT_BLOCK / NEAR_TEXT / section attribution / lazy-seed / interleaved-picture regression. |
| `tests/integration/test_docling_anchors_smoke.py` | Extend with §7 integration assertions. |
| `tests/fixtures/docling_anchors/*.json` | Reuse existing four fixtures where possible. Add one new fixture if needed for interleaved captioned/uncaptioned pictures or pre-heading anchored content. |
| `VERIFICATION_CHECKLIST.md` | New entity + relationship entries, migration note. |

## 9. What v2.2 drops vs v1

| v1 (invalid) | v2.2 (this doc) |
|---|---|
| New `/extract-structure` endpoint on docling-graph | None — existing walker |
| New manifest field `extraction_method: structural` | None |
| Worker routing change in `_call_extract_pass` | None |
| New `DocumentStructurePass` Pydantic pass template | None |
| Manifest ordering change (document_structure becomes pass 1) | None |
| `CONTAINS_SECTION` / `CONTAINS_FIGURE` / `CONTAINS_TABLE` / `CONTAINS_IMAGE` / `DESCRIBES` relationships | Reuses existing HAS_* convention; only HAS_IMAGE + NEAR_TEXT are new |
| `primary_entity_types: [DOCUMENT, SECTION, FIGURE, TABLE, IMAGE]` on pass definition | N/A — not a pass |
| `upstream_refs` flowing to downstream passes E001..E00N | Already happens via merger — no change needed for this scope |
| Pydantic-edge-helper typed edges on pass root | Walker emits `MergedEdgeRecord` directly, consistent with existing code |
