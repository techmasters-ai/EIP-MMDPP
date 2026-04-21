# Document Structure Extraction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `app/services/docling_anchors.py` to emit IMAGE + TEXT_BLOCK entities, section-level attribution, and surrounding-text NEAR_TEXT edges, and align the air_defense_v3 ontology (enum + matrix + registry) with what the walker already emits.

**Architecture:** No new service endpoint, no new manifest pass, no worker-routing change. All work extends the existing deterministic `docling_anchors.walk()` called from the `derive_document_anchors` Celery task at `app/workers/pipeline.py:4316`. Design: `docs/plans/2026-04-21-document-structure-pass-design.md` (v2.3).

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy (read-only here), Docling + docling-core (`DoclingDocument.iterate_items()`, `PictureItem`, `TableItem`, `SectionHeaderItem`), pytest, pytest-style JSON fixtures at `tests/fixtures/docling_anchors/`.

**Branch:** `feat/pydantic-ontology-ssot` (continues current feature branch).

---

## File map

| File | Change kind |
|---|---|
| `ontology_bundles/air_defense_v3/relationships.py` | Modify — add 6 enum members + 6 descriptors |
| `ontology_bundles/air_defense_v3/validation_matrix.py` | Modify — add 10 triples |
| `ontology_bundles/air_defense_v3/entities.py` | Modify — add ImageEntity + TextBlockEntity, storage_key fields, ALL_ENTITIES registration |
| `app/services/docling_anchors.py` | Modify — walker extension (sections 4.1–4.6 of design) |
| `app/workers/pipeline.py` | Modify — pass `source_storage_key` at `:4316` |
| `tests/unit/test_docling_anchor_walker.py` | Modify — extend with new scenarios (TDD, interleaved with each walker change) |
| `tests/integration/test_docling_anchors_smoke.py` | Modify — smoke assertions for IMAGE + TEXT_BLOCK + NEAR_TEXT |
| `tests/fixtures/docling_anchors/` | Add — one new fixture for interleaved captioned/uncaptioned pictures |
| `VERIFICATION_CHECKLIST.md` | Modify — document new entities + relationships + re-ingest note |
| `notebooks/ingest_walkthrough.ipynb` | Modify — surface IMAGE + TEXT_BLOCK in inspection cells |
| `notebooks/raw_libraries_walkthrough.ipynb` | Modify — mirror notebook-side walker logic |

Task order: **Chunk 1** (Task 1 + Task 2 — independent, can land in any order) → **Chunk 2** (Task 3 walker, blocked by 1+2) → **Chunk 3** (Task 5 tests+checklist, blocked by 3) → **Chunk 4** (Task 6 notebooks, blocked by 5).

---

## Chunk 1: Ontology additions (Tasks 1 + 2)

These two tasks have no dependency on each other. If executed in parallel, each gets its own commit.

### Task 1: Relationships + validation_matrix

**Files:**
- Modify: `ontology_bundles/air_defense_v3/relationships.py:20-76` (enum) + `:128+` (`_STATIC_RELATIONSHIP_METADATA` list)
- Modify: `ontology_bundles/air_defense_v3/validation_matrix.py:26+` (`VALIDATION_MATRIX` frozenset body)
- Test: `tests/unit/test_relationships_and_matrix_anchors.py` *(create — small new file covering just the additions)*

- [ ] **Step 1.1: Write the failing test**

Create `tests/unit/test_relationships_and_matrix_anchors.py`:

```python
"""Enum + matrix additions for the docling-anchor set.

See docs/plans/2026-04-21-document-structure-pass-design.md §3.
"""
from __future__ import annotations

from ontology_bundles.air_defense_v3.relationships import (
    RelationshipType,
    RELATIONSHIP_METADATA,
)
from ontology_bundles.air_defense_v3.validation_matrix import VALIDATION_MATRIX


def test_anchor_enum_members_present():
    for name in (
        "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE",
        "CHILD_OF", "HAS_IMAGE", "NEAR_TEXT",
    ):
        assert hasattr(RelationshipType, name), f"RelationshipType.{name} missing"


def test_anchor_metadata_descriptors_present():
    for name in (
        "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE",
        "CHILD_OF", "HAS_IMAGE", "NEAR_TEXT",
    ):
        assert RelationshipType[name] in RELATIONSHIP_METADATA, (
            f"RELATIONSHIP_METADATA missing descriptor for {name}"
        )


def test_validation_matrix_has_10_new_triples():
    expected = {
        ("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
        ("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
        ("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
        ("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),
        ("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
        ("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
        ("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),
        ("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),
        ("FIGURE",   RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
        ("IMAGE",    RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    }
    assert expected <= VALIDATION_MATRIX, (
        f"Missing triples: {expected - VALIDATION_MATRIX}"
    )
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_relationships_and_matrix_anchors.py -v
```

Expected: `test_anchor_enum_members_present` FAILs on first missing member; downstream tests error-out or fail.

- [ ] **Step 1.3: Add 6 enum members to `RelationshipType`**

Edit `ontology_bundles/air_defense_v3/relationships.py` after line 54 (end of current `HAS_*` block) — add six members in enum-style alphabetical order where it fits. Concretely, insert into the enum body so all HAS_* stay grouped:

```python
    HAS_FIGURE = "HAS_FIGURE"
    HAS_IMAGE = "HAS_IMAGE"
    HAS_SECTION = "HAS_SECTION"
    HAS_TABLE = "HAS_TABLE"
```

And add these two wherever they fit alphabetically in the enum (CHILD_OF goes near top-of-alphabet members, NEAR_TEXT near bottom):

```python
    CHILD_OF = "CHILD_OF"
    NEAR_TEXT = "NEAR_TEXT"
```

- [ ] **Step 1.4: Add 6 descriptor dicts to `_STATIC_RELATIONSHIP_METADATA`**

Same file, append to the list at `:128+` (before the list-close `]`):

```python
    {"name": "HAS_SECTION","label": "Has Section","description": "Document or parent contains a section","source_type": None,"target_type": "SECTION","cardinality": "one_to_many"},
    {"name": "HAS_FIGURE", "label": "Has Figure", "description": "Document or section contains a captioned figure","source_type": None,"target_type": "FIGURE","cardinality": "one_to_many"},
    {"name": "HAS_TABLE",  "label": "Has Table",  "description": "Document or section contains a table","source_type": None,"target_type": "TABLE","cardinality": "one_to_many"},
    {"name": "CHILD_OF",   "label": "Child Of",   "description": "Hierarchical containment (e.g. sub-section within parent section)","source_type": None,"target_type": None,"cardinality": "many_to_one"},
    {"name": "HAS_IMAGE",  "label": "Has Image",  "description": "Document or section contains an uncaptioned image or embedded picture","source_type": None,"target_type": "IMAGE","cardinality": "one_to_many"},
    {"name": "NEAR_TEXT",  "label": "Near Text",  "description": "Figure or image appears near a text block in reading order","source_type": None,"target_type": "TEXT_BLOCK","cardinality": "one_to_many"},
```

- [ ] **Step 1.5: Add 10 triples to `VALIDATION_MATRIX`**

Edit `ontology_bundles/air_defense_v3/validation_matrix.py`, inserting into the frozenset at `:26+` in alphabetical order by source_type (the existing set is alphabetical by source):

```python
    ("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
    ("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
    ("FIGURE",   RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("IMAGE",    RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),
    ("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
```

- [ ] **Step 1.6: Run tests to verify they pass**

```bash
pytest tests/unit/test_relationships_and_matrix_anchors.py -v
```

Expected: 3 PASS. Also run the full unit suite to verify no regression:

```bash
pytest tests/unit/ -x -q
```

Expected: all green.

- [ ] **Step 1.7: Commit**

```bash
git add ontology_bundles/air_defense_v3/relationships.py \
        ontology_bundles/air_defense_v3/validation_matrix.py \
        tests/unit/test_relationships_and_matrix_anchors.py
git commit -m "feat(ontology): register anchor relationships (HAS_SECTION/FIGURE/TABLE/IMAGE, CHILD_OF, NEAR_TEXT)

Walker emits HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF as raw strings
today but they were absent from RelationshipType. Adds the four missing
anchor relations plus HAS_IMAGE + NEAR_TEXT for the incoming IMAGE +
TEXT_BLOCK extraction work. _STATIC_RELATIONSHIP_METADATA + VALIDATION_MATRIX
updated in lockstep. See docs/plans/2026-04-21-document-structure-pass-design.md §3."
```

---

### Task 2: Schemas — ImageEntity + TextBlockEntity + storage_key

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py` (add two new classes after `TableEntity` at `:136`; register in `ALL_ENTITIES` at `:1285`; add `storage_key` field to `DocumentEntity:64-97` and `FigureEntity:112-124`)
- Test: `tests/unit/test_entity_schemas_anchors.py` *(create)*

- [ ] **Step 2.1: Write failing tests**

Create `tests/unit/test_entity_schemas_anchors.py`:

```python
"""Pydantic entity additions for the docling-anchor set.

See docs/plans/2026-04-21-document-structure-pass-design.md §2.
"""
from __future__ import annotations

import pytest

from ontology_bundles.air_defense_v3.entities import (
    ALL_ENTITIES,
    DocumentEntity,
    FigureEntity,
)


def test_image_entity_imports_and_validates():
    from ontology_bundles.air_defense_v3.entities import ImageEntity
    img = ImageEntity(
        image_ref="#/pictures/7",
        document_id="doc-uuid-1",
        page=3,
        caption=None,
        mime_type="image/png",
        storage_key=None,
        bbox={"l": 0, "t": 0, "r": 100, "b": 100, "page": 3, "coord_origin": "TOPLEFT"},
        image_role="INLINE_IMAGE",
        confidence=1.0,
    )
    assert img.image_ref == "#/pictures/7"
    assert img.model_config["ontology_name"] == "IMAGE"
    assert img.model_config["graph_id_fields"] == ["image_ref"]
    assert img.model_config["identity_scope"] == "document"
    assert img.model_config["is_entity"] is True


def test_text_block_entity_imports_and_validates():
    from ontology_bundles.air_defense_v3.entities import TextBlockEntity
    tb = TextBlockEntity(
        text_ref="#/texts/42",
        document_id="doc-uuid-1",
        text="Some paragraph text",
        label="TEXT",
        page=3,
        confidence=1.0,
    )
    assert tb.text_ref == "#/texts/42"
    assert tb.model_config["ontology_name"] == "TEXT_BLOCK"
    assert tb.model_config["graph_id_fields"] == ["text_ref"]


def test_image_and_text_block_registered_in_ALL_ENTITIES():
    from ontology_bundles.air_defense_v3.entities import ImageEntity, TextBlockEntity
    assert ALL_ENTITIES.get("IMAGE") is ImageEntity
    assert ALL_ENTITIES.get("TEXT_BLOCK") is TextBlockEntity


def test_document_entity_has_storage_key_field():
    doc = DocumentEntity(document_number="TM 9-1425-386-12", storage_key="minio-key-abc")
    assert doc.storage_key == "minio-key-abc"


def test_figure_entity_has_storage_key_field():
    fig = FigureEntity(figure_ref="#/pictures/0", storage_key=None)
    # storage_key always null in this change — see design v2.3 §4.4.
    assert fig.storage_key is None
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_entity_schemas_anchors.py -v
```

Expected: ImportError on `ImageEntity` / `TextBlockEntity`; attribute errors on the two `storage_key` tests.

- [ ] **Step 2.3: Add `ImageEntity` and `TextBlockEntity` in entities.py**

Insert after `TableEntity` (at `entities.py:136`, just before the "Layer 2" comment block):

```python
class ImageEntity(BaseModel):
    """Image — An uncaptioned picture or embedded image within a document.

    Distinguished from FigureEntity by the absence of a "Figure N"-style
    caption: embedded logos, inline diagrams without labels, decorative
    photos. See docs/plans/2026-04-21-document-structure-pass-design.md §2.
    """
    model_config = ConfigDict(
        ontology_name="IMAGE",
        graph_id_fields=["image_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )

    image_ref: str = Field(..., description="Document-scoped self_ref of the picture (docs:17235 R16-compliant identity)", examples=["#/pictures/7", "#/pictures/12"])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this image belongs to")
    page: Optional[int] = Field(default=None, description="Page number where the image appears", examples=[1, 3])
    caption: Optional[str] = Field(default=None, description="Caption text when present, else None", examples=["Installation overview"])
    mime_type: Optional[str] = Field(default=None, description="MIME type of the backing image asset", examples=["image/png", "image/jpeg"])
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the picture bytes. Always null in the initial change; populated once Artifact.self_ref plumbing lands.")
    bbox: Optional[dict] = Field(default=None, description="Bounding box dict {l, t, r, b, page, coord_origin} from Docling provenance")
    image_role: Optional[str] = Field(
        default=None,
        description="Role heuristic derived from page position + caption",
        json_schema_extra={"enum": ["HEADER_LOGO", "INLINE_IMAGE", "UNCAPTIONED_FIGURE"]},
    )
    confidence: Optional[float] = Field(default=1.0, description="Extraction confidence, 0–1. Anchor walker always emits 1.0.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})


class TextBlockEntity(BaseModel):
    """TextBlock — A body-text paragraph emitted as neighbor context for FIGURE/IMAGE.

    Lazily emitted by the anchor walker: a TEXT_BLOCK only appears when at
    least one IMAGE or FIGURE declared it as a NEAR_TEXT neighbor. See
    docs/plans/2026-04-21-document-structure-pass-design.md §4.3.
    """
    model_config = ConfigDict(
        ontology_name="TEXT_BLOCK",
        graph_id_fields=["text_ref"],
        identity_scope="document",
        dodaf_parent="DocumentResource",
        is_entity=True,
    )

    text_ref: str = Field(..., description="Document-scoped self_ref of the text item (docs:17235 R16-compliant identity)", examples=["#/texts/42", "#/texts/237"])
    document_id: Optional[str] = Field(default=None, description="Internal document UUID this text block belongs to")
    text: Optional[str] = Field(default=None, description="Rendered text content, truncated to 500 characters")
    label: Optional[str] = Field(default=None, description="Docling label (TEXT, PARAGRAPH, LIST_ITEM)", examples=["TEXT", "PARAGRAPH", "LIST_ITEM"])
    page: Optional[int] = Field(default=None, description="Page number where the text appears", examples=[3])
    confidence: Optional[float] = Field(default=1.0, description="Extraction confidence, 0–1. Anchor walker always emits 1.0.", ge=0.0, le=1.0, json_schema_extra={"system_field": True})
```

- [ ] **Step 2.4: Add `storage_key` to `DocumentEntity` and `FigureEntity`**

In `DocumentEntity` (at `entities.py:64-97`), add a new field — the Layer 1 convention puts production-metadata fields before `confidence`. Add after `publication_date`:

```python
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the source file in the documents bucket")
```

In `FigureEntity` (at `entities.py:112-124`), add after `figure_type`:

```python
    storage_key: Optional[str] = Field(default=None, description="MinIO object key for the figure image. Always null in the initial change; populated once Artifact.self_ref plumbing lands.")
```

- [ ] **Step 2.5: Register both new classes in `ALL_ENTITIES`**

Edit `entities.py:1285-1330`, insert the two entries in the Layer 1 section (the existing four — DOCUMENT / SECTION / FIGURE / TABLE — live at the top of the dict):

```python
ALL_ENTITIES: dict[str, type[BaseModel]] = {
    "DOCUMENT": DocumentEntity,
    "SECTION": SectionEntity,
    "FIGURE": FigureEntity,
    "TABLE": TableEntity,
    "IMAGE": ImageEntity,                    # NEW
    "TEXT_BLOCK": TextBlockEntity,           # NEW
    ...existing entries unchanged...
}
```

The `model_rebuild()` loop at `:1333-1334` automatically runs for both new classes.

- [ ] **Step 2.6: Run tests to verify they pass**

```bash
pytest tests/unit/test_entity_schemas_anchors.py -v
```

Expected: 5 PASS.

Full regression:

```bash
pytest tests/unit/ -x -q
```

Expected: all green.

- [ ] **Step 2.7: Commit**

```bash
git add ontology_bundles/air_defense_v3/entities.py tests/unit/test_entity_schemas_anchors.py
git commit -m "feat(schemas): add ImageEntity + TextBlockEntity; storage_key on DOCUMENT/FIGURE

Two new Pydantic entities for the docling-anchor walker:
- ImageEntity (uncaptioned pictures; image_role enum: HEADER_LOGO /
  INLINE_IMAGE / UNCAPTIONED_FIGURE)
- TextBlockEntity (lazily emitted neighbors of FIGURE/IMAGE)

Adds Optional[str] storage_key to DocumentEntity + FigureEntity.
DocumentEntity.storage_key will be populated from the SQL row in the
walker-extension task; FigureEntity/ImageEntity.storage_key stay null
until Artifact.self_ref plumbing lands (out of scope here).

Both new classes registered in ALL_ENTITIES so model_rebuild() covers
them. See docs/plans/2026-04-21-document-structure-pass-design.md §2."
```

---

## Chunk 2: Walker extension (Task 3)

**Blocked by:** Chunk 1 (Tasks 1 + 2) — both must be committed before this work starts; the walker imports `ImageEntity` + `TextBlockEntity` and emits relationships that reference the new enum members.

**Files:**
- Modify: `app/services/docling_anchors.py` (see substeps — single file, many small additions)
- Modify: `app/workers/pipeline.py:4316` (one line at the call-site)
- Test: `tests/unit/test_docling_anchor_walker.py` (add new test functions, TDD-paired with each substep)
- Fixture: `tests/fixtures/docling_anchors/interleaved_pictures.json` (create for the 3h regression test)

Each substep below is self-contained: write test → run fail → implement → run pass → commit. The substeps are ordered so each builds on the previous.

### Task 3a: Helper — `_ensure_root_section`, and rewrite zero-heading fallback as end-of-traversal call

- [ ] **Step 3a.1: Write failing test**

Add to `tests/unit/test_docling_anchor_walker.py`:

```python
def test_walker_zero_headings_zero_pictures_still_emits_root_section(ontology):
    """End-of-traversal fallback — covers headingless docs with nothing
    anchored either. Must still produce SECTION(section_number="0")."""
    doc = _build_doc(
        title=None,
        sections=[],
        texts=["Body paragraph with no heading context."],
        pictures=[], tables=[],
    )
    merged = walk(doc.model_dump(mode="json"), "doc-1", "run-1", ontology)
    numbers = _collect_section_numbers(merged)
    assert numbers == ["0"], f"expected ['0'], got {numbers}"
```

(If helpers like `_build_doc` or `ontology` fixture aren't yet present for this shape of doc, reuse the existing `test_walker_empty_structure_emits_fallback_section` pattern as a template.)

- [ ] **Step 3a.2: Run test (should still pass with current code)**

```bash
pytest tests/unit/test_docling_anchor_walker.py::test_walker_zero_headings_zero_pictures_still_emits_root_section -v
```

Expected: PASS against current walker (the existing fallback at lines 210–216 handles this). We're writing the test FIRST to lock behavior before refactoring.

- [ ] **Step 3a.3: Refactor walker to a single `_ensure_root_section` helper**

In `app/services/docling_anchors.py`, inside `walk()` above the `iterate_items()` loop, add:

```python
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
```

Delete the existing all-headings-absent fallback at lines 210-216:

```python
    # --- REMOVE these lines ---
    if not section_by_path:
        section_by_path[("",)] = SectionEntity(
            section_number="0",
            heading=None,
            section_path=None,
        )
```

And replace with an explicit end-of-traversal call:

```python
    # End-of-traversal fallback — covers zero-headings AND zero-anchored-content.
    if not section_by_path:
        _ensure_root_section()
```

NOTE: the old fallback keyed on `("",)` (tuple of empty string); the helper keys on `()` (empty tuple). This aligns SECTION attribution lookups in §3c below. Tests that relied on the `("",)` key would need updating — search for it:

```bash
grep -rn '("",)' tests/
```

- [ ] **Step 3a.4: Run test to verify it passes with the new helper**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v
```

Expected: new test PASSes; all existing walker tests still PASS.

- [ ] **Step 3a.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "refactor(anchors): _ensure_root_section helper; end-of-traversal fallback

Single synthetic-root-section code path that both the zero-headings
fallback and the upcoming pre-heading picture attribution use. Keys on
empty-tuple () instead of ('',) so downstream SECTION attribution
lookups (§3c) match the walker's section_stack-derived keys. Design §4.1a."
```

### Task 3b: Track pic/tbl → section during iterate_items + lazy-seed root section when pre-heading anchored content appears

- [ ] **Step 3b.1: Write failing test**

Add to `tests/unit/test_docling_anchor_walker.py`:

```python
from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem
from docling_core.types.doc.document import ProvenanceItem, BoundingBox

def test_walker_pre_heading_picture_lazy_seeds_root_section(ontology):
    """A picture appearing before the first heading (in a doc that HAS
    headings later) must attach to the synthetic section_number='0'.
    Without lazy-seeding, section_by_path.get(()) returns None and the
    SECTION→IMAGE edge is silently dropped. Design §4.1a."""
    doc = DoclingDocument(name="pre_heading_pic")
    doc.add_picture(
        image=None,
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 0)),
    )
    doc.add_heading("Section 1", level=1)
    merged = walk(doc.model_dump(mode="json"), "doc-1", "run-1", ontology)

    sections = _sections_by_number(merged)
    assert "0" in sections, "root fallback SECTION(section_number='0') missing"
    # Also assert the picture's SECTION→IMAGE edge exists, pointing at "0".
    section_image_edges = [
        e for e in merged.edges
        if e.rel_type == "HAS_IMAGE"
        and e.from_identity.entity_type == "SECTION"
    ]
    assert len(section_image_edges) == 1
    # Tuple key for root section is ().
```

- [ ] **Step 3b.2: Run test to verify it fails**

```bash
pytest tests/unit/test_docling_anchor_walker.py::test_walker_pre_heading_picture_lazy_seeds_root_section -v
```

Expected: FAIL — `IMAGE` entity type doesn't exist yet in walker output, so assertion fails.

NOTE: this test's final assertion depends on Task 3c (picture split) + 3g (SECTION→IMAGE edge). Mark it `xfail` if you want to commit 3b independently, or keep it red and complete 3c + 3g before running this green. Recommendation: keep red, run green after 3g lands.

- [ ] **Step 3b.3: Add pic/tbl tracking dicts + lazy-seed call inside `iterate_items()` loop**

In `docling_anchors.py::walk()`, before the `iterate_items()` loop, initialize four dicts:

```python
    # §4.1 — per-picture/table section-stack + reading-order capture.
    pic_to_section:     dict[str, tuple[str, ...]] = {}
    tbl_to_section:     dict[str, tuple[str, ...]] = {}
    pic_to_order_index: dict[str, int] = {}
    tbl_to_order_index: dict[str, int] = {}
    all_items_in_order: list = []
```

Replace the `for item, tree_depth in docling_doc.iterate_items():` line with:

```python
    from docling_core.types.doc import PictureItem, TableItem
    for order_index, (item, tree_depth) in enumerate(docling_doc.iterate_items()):
        all_items_in_order.append(item)
```

Inside the loop, at the end (after the existing section_stack logic + `_register_section(path_tuple)`), add:

```python
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
```

- [ ] **Step 3b.4: Run existing walker tests to verify no regression**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "not pre_heading"
```

Expected: all existing tests PASS. The `pre_heading` test stays red — it needs 3c + 3g to fully pass.

- [ ] **Step 3b.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): per-picture/table section tracking + pre-heading lazy seed

During iterate_items() loop, record each PictureItem / TableItem
self_ref against the current section_stack (as a tuple) and its
order index. Call _ensure_root_section() when anchored content
appears before any heading. Enables SECTION attribution for
FIGURE/TABLE/IMAGE in subsequent substeps. Design §4.1, §4.1a."
```

### Task 3c: Split pictures into FIGURE vs IMAGE; populate `pic_entities` aligned list

- [ ] **Step 3c.1: Write failing tests**

Add to `tests/unit/test_docling_anchor_walker.py`:

```python
def test_walker_captioned_figure_stays_figure(ontology):
    """Regression: a picture whose caption starts with 'Figure N' still
    emits a FigureEntity with the caption label."""
    # Build a doc with one picture + a caption text_item referencing it.
    # (Use existing test_walker_figure_count_and_ref as a template for the
    # caption wiring.)
    ...
    merged = walk(..., ontology)
    figures = [e for e in merged.entities if e.identity.entity_type == "FIGURE"]
    assert len(figures) == 1
    assert figures[0].properties["figure_label"] == "Figure 3-12"


def test_walker_uncaptioned_picture_becomes_image(ontology):
    """A picture with no Figure-N caption emits an ImageEntity."""
    ...
    merged = walk(..., ontology)
    images = [e for e in merged.entities if e.identity.entity_type == "IMAGE"]
    assert len(images) == 1
    assert images[0].identity.identity_tuple == ("#/pictures/0",)
    # storage_key always None in this change — see design §4.4.
    assert images[0].properties.get("storage_key") is None


def test_walker_interleaved_captioned_uncaptioned(ontology):
    """Captioned FIGURE + uncaptioned IMAGE interleaved in document
    order — the pic_entities parallel list must preserve doc order so
    §3e NEAR_TEXT lookups hit the right picture. Design §4.3 zip fix."""
    # Three pictures: captioned, uncaptioned, captioned.
    ...
    merged = walk(..., ontology)
    figures = [e for e in merged.entities if e.identity.entity_type == "FIGURE"]
    images  = [e for e in merged.entities if e.identity.entity_type == "IMAGE"]
    assert len(figures) == 2
    assert len(images) == 1
    # Ordering within each bucket reflects doc order.
```

- [ ] **Step 3c.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "caption or uncaptioned or interleaved"
```

Expected: 3 FAIL — walker doesn't emit IMAGE yet.

- [ ] **Step 3c.3: Add picture-split logic in `docling_anchors.py`**

Import ImageEntity at the top:

```python
from ontology_bundles.air_defense_v3.entities import (
    DocumentEntity,
    FigureEntity,
    ImageEntity,     # NEW
    SectionEntity,
    TableEntity,
)
```

Replace the existing picture loop (`docling_anchors.py:219-226`) with:

```python
    figures: list[FigureEntity] = []
    images:  list[ImageEntity]  = []
    pic_entities: list = []   # parallel to docling_doc.pictures for §3e NEAR_TEXT

    for pic in docling_doc.pictures:
        label = _caption_label(pic)
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
                caption=_caption_label(pic),  # caption text even if not "Figure N"
                storage_key=None,
                image_role=_classify_image_role(pic, docling_doc),
            )
            images.append(entity)
        pic_entities.append(entity)
```

Add the role classifier helper near the top of the file (next to `_caption_label`):

```python
def _classify_image_role(pic, docling_doc) -> str:
    """Heuristic role assignment (§4.2). Returns HEADER_LOGO, INLINE_IMAGE,
    or UNCAPTIONED_FIGURE."""
    label = _caption_label(pic)
    # Page geometry
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

    # Rule 2 — caption present but not "Figure N" prefix.
    if label:
        return "UNCAPTIONED_FIGURE"
    # Rule 3.
    return "INLINE_IMAGE"
```

- [ ] **Step 3c.4: Extend entity_models assembly at end of walk()**

At `docling_anchors.py:301-306`, the entity assembly currently does:

```python
    entity_models.extend(figures)
    entity_models.extend(tables)
```

Add images to the list — placed between figures and tables:

```python
    entity_models.extend(figures)
    entity_models.extend(images)     # NEW
    entity_models.extend(tables)
```

(TEXT_BLOCKs are added in §3e.)

- [ ] **Step 3c.5: Run tests to verify they pass**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "caption or uncaptioned or interleaved"
```

Expected: 3 PASS.

Full walker suite:

```bash
pytest tests/unit/test_docling_anchor_walker.py -v
```

Expected: all pass (the `pre_heading` test from 3b is still red until 3g; mark xfail if desired).

- [ ] **Step 3c.6: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): split FIGURE vs IMAGE based on caption; HEADER_LOGO heuristic

Pictures with 'Figure N'-style captions still emit FigureEntity.
Uncaptioned pictures emit ImageEntity with image_role chosen from:
- HEADER_LOGO: page-1 small-top-bbox (area < 10% page area)
- UNCAPTIONED_FIGURE: has caption but no Figure-N prefix
- INLINE_IMAGE: everything else

pic_entities parallel list preserves docling_doc.pictures order so
downstream NEAR_TEXT attachment (§3e) zips correctly. Design §4.2."
```

### Task 3d: `_neighbors()` reading-order helper

- [ ] **Step 3d.1: Write failing test**

```python
def test_neighbors_window_skips_non_text_items(ontology):
    """_neighbors returns up to 2 text items before + 2 after target,
    skipping section headers and captions of other pictures/tables."""
    from app.services.docling_anchors import _neighbors, _is_valid_near_text
    # Build an items list: [text, section_header, text, picture_target,
    #                       caption_text, text, picture_other, text]
    # Target at index 3. Expected: [text@0, text@2, text@5, text@7].
    ...
```

(The exact fixture composition depends on which `_build_doc`-style helpers exist in the test file; follow the pattern used by `test_walker_three_level_hierarchy` for heading/text sequencing.)

- [ ] **Step 3d.2: Run test to verify it fails**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "neighbors_window"
```

Expected: FAIL — `_neighbors` not defined.

- [ ] **Step 3d.3: Add `_neighbors` + `_is_valid_near_text` helpers in docling_anchors.py**

```python
from docling_core.types.doc import DocItemLabel, TextItem


def _is_valid_near_text(item, captions_linked: set[str]) -> bool:
    """Accept TextItem whose label is body text (not section/title/caption).
    captions_linked is the set of self_refs already used as picture/table
    captions; those must be excluded from neighbor windows."""
    if not isinstance(item, TextItem):
        return False
    label = getattr(item, "label", None)
    if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.CAPTION):
        return False
    if getattr(item, "self_ref", None) in captions_linked:
        return False
    return True


def _neighbors(target_order: int, items: list, captions_linked: set[str], window: int = 2) -> list:
    """Return up to `window` valid-near-text items before + `window` after
    target_order, in reading order. Design §4.3."""
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
```

Additionally, before calling `_neighbors`, compute the `captions_linked` set:

```python
    # Collect self_refs of all caption text items already owned by pictures/tables.
    captions_linked: set[str] = set()
    for item in list(docling_doc.pictures) + list(docling_doc.tables):
        captions = getattr(item, "captions", None) or []
        for cap in captions:
            ref = getattr(cap, "cref", None)
            if isinstance(ref, str):
                captions_linked.add(ref)
```

- [ ] **Step 3d.4: Run test to verify it passes**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "neighbors_window"
```

Expected: PASS.

- [ ] **Step 3d.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): _neighbors helper — reading-order ±N text blocks

Skips section headers, titles, and captions already linked to other
pictures/tables. Foundation for TEXT_BLOCK lazy emission in §3e."
```

### Task 3e: Lazy TEXT_BLOCK emission + NEAR_TEXT edges (attached to FIGURE + IMAGE)

- [ ] **Step 3e.1: Write failing tests**

```python
def test_walker_near_text_window_on_figure(ontology):
    """Each FIGURE gets up to 4 TextBlockEntity children via NEAR_TEXT."""
    ...
    merged = walk(..., ontology)
    fig_identity = next(e.identity for e in merged.entities if e.identity.entity_type == "FIGURE")
    near_text_edges = [e for e in merged.edges
                       if e.rel_type == "NEAR_TEXT" and e.from_identity == fig_identity]
    assert len(near_text_edges) >= 1
    assert all(e.to_identity.entity_type == "TEXT_BLOCK" for e in near_text_edges)


def test_walker_near_text_dedup_across_pictures(ontology):
    """A text block shared by two pictures → one TextBlockEntity,
    two NEAR_TEXT edges."""
    ...
    tbs = [e for e in merged.entities if e.identity.entity_type == "TEXT_BLOCK"]
    shared_tb = [t for t in tbs if t.identity.identity_tuple == ("#/texts/5",)]
    assert len(shared_tb) == 1
    edges = [e for e in merged.edges if e.to_identity == shared_tb[0].identity]
    assert len(edges) == 2
```

- [ ] **Step 3e.2: Run tests to verify they fail**

Expected: no TEXT_BLOCK entities in output.

- [ ] **Step 3e.3: Add TEXT_BLOCK emission + NEAR_TEXT edges in docling_anchors.py**

After the picture-split loop (§3c), before the edge-building block (line 237):

```python
    # §4.3 — lazy TEXT_BLOCK emission + NEAR_TEXT edges.
    text_blocks_by_ref: dict[str, TextBlockEntity] = {}  # dedup key: text_ref
    near_text_pairs: list[tuple] = []  # (parent_entity, text_block_entity)

    # pic_entities is built 1:1 with docling_doc.pictures in §3c.
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
                prov_page = None
                try:
                    prov = getattr(text_item, "prov", None)
                    if prov:
                        prov_page = prov[0].page_no
                except (AttributeError, IndexError):
                    pass
                tb = TextBlockEntity(
                    text_ref=self_ref,
                    text=text,
                    label=label_value,
                    page=prov_page,
                )
                text_blocks_by_ref[self_ref] = tb
            near_text_pairs.append((entity, tb))
```

Import `TextBlockEntity` at the top:

```python
from ontology_bundles.air_defense_v3.entities import (
    DocumentEntity,
    FigureEntity,
    ImageEntity,
    SectionEntity,
    TableEntity,
    TextBlockEntity,   # NEW
)
```

Extend the edge-building block to emit NEAR_TEXT edges:

```python
    # §4.3 — NEAR_TEXT edges
    for parent_entity, tb in near_text_pairs:
        edges.append(MergedEdgeRecord(
            from_identity=_identity(parent_entity),
            to_identity=_identity(tb),
            rel_type="NEAR_TEXT",
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))
```

Extend entity assembly:

```python
    entity_models.extend(text_blocks_by_ref.values())  # NEW
```

- [ ] **Step 3e.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "near_text"
```

Expected: 2 PASS.

- [ ] **Step 3e.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): TEXT_BLOCK + NEAR_TEXT emission for FIGURE and IMAGE

Lazily emits a TextBlockEntity per neighbor referenced by at least one
IMAGE or FIGURE. Dedup by text_ref: a block shared between two pictures
is one vertex with two NEAR_TEXT edges. Design §4.3."
```

### Task 3f: SECTION→FIGURE/TABLE/IMAGE + DOCUMENT→IMAGE edges

- [ ] **Step 3f.1: Write failing tests**

```python
def test_walker_section_has_figure_edge(ontology):
    ...  # Figure in "Section 1.1" → edge SECTION(1.1) → FIGURE

def test_walker_section_has_table_edge(ontology):
    ...  # Table in "Section 2" → edge

def test_walker_section_has_image_edge(ontology):
    ...  # Uncaptioned picture in "Section 3" → edge

def test_walker_document_has_image_edge_when_designator_present(ontology):
    ...  # DOCUMENT exists + at least one IMAGE → DOCUMENT→IMAGE edge
```

- [ ] **Step 3f.2: Run tests to verify they fail**

Expected: edges absent from walker output.

- [ ] **Step 3f.3: Add SECTION-level + DOCUMENT→IMAGE edges**

In the edge-building block, after the existing DOCUMENT→FIGURE/TABLE block:

```python
    # DOCUMENT → IMAGE (only when doc_entity emitted) — matches existing HAS_* pattern.
    if doc_entity is not None:
        for img in images:
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity,
                to_identity=_identity(img),
                rel_type="HAS_IMAGE",
                confidence=1.0,
                pass_origins={"document_anchors"},
            ))

    # SECTION → FIGURE/TABLE/IMAGE — unconditional on DOCUMENT.
    # §4.5: attribute by the pic_to_section / tbl_to_section lookup.
    for pic, entity in zip(docling_doc.pictures, pic_entities, strict=True):
        section_key = pic_to_section.get(pic.self_ref, ())
        section = section_by_path.get(section_key)
        if section is None:
            continue
        rel = "HAS_FIGURE" if isinstance(entity, FigureEntity) else "HAS_IMAGE"
        edges.append(MergedEdgeRecord(
            from_identity=_identity(section),
            to_identity=_identity(entity),
            rel_type=rel,
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))

    for tbl, table_entity in zip(docling_doc.tables, tables, strict=True):
        section_key = tbl_to_section.get(tbl.self_ref, ())
        section = section_by_path.get(section_key)
        if section is None:
            continue
        edges.append(MergedEdgeRecord(
            from_identity=_identity(section),
            to_identity=_identity(table_entity),
            rel_type="HAS_TABLE",
            confidence=1.0,
            pass_origins={"document_anchors"},
        ))
```

- [ ] **Step 3f.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v
```

Expected: all tests PASS (including the `pre_heading` test from 3b.1, which can now go green).

- [ ] **Step 3f.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): SECTION→FIGURE/TABLE/IMAGE + DOCUMENT→IMAGE HAS_* edges

Section-level attribution via the pic_to_section / tbl_to_section
lookups built in §3b. Pre-heading anchored content lands on the
synthetic root section ('0') seeded by _ensure_root_section.
Design §4.5."
```

### Task 3g: `source_storage_key` kwarg + DocumentEntity population

- [ ] **Step 3g.1: Write failing test**

```python
def test_walker_populates_document_storage_key_when_kwarg_passed(ontology):
    """source_storage_key kwarg populates DocumentEntity.storage_key."""
    doc = _build_doc(title="TM 9-1425-386-12", ...)
    merged = walk(
        doc.model_dump(mode="json"),
        "doc-1",
        "run-1",
        ontology,
        source_storage_key="docs/abc.pdf",
    )
    docs = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert len(docs) == 1
    assert docs[0].properties["storage_key"] == "docs/abc.pdf"


def test_walker_document_storage_key_null_when_kwarg_absent(ontology):
    """No kwarg → DocumentEntity.storage_key is None."""
    doc = _build_doc(title="TM 9-1425-386-12", ...)
    merged = walk(doc.model_dump(mode="json"), "doc-1", "run-1", ontology)
    docs = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert docs[0].properties.get("storage_key") is None
```

- [ ] **Step 3g.2: Run tests to verify they fail**

Expected: TypeError on unexpected kwarg, or storage_key absent from properties.

- [ ] **Step 3g.3: Add `source_storage_key` kwarg + use it**

Edit `walk()` signature:

```python
def walk(
    docling_doc_json: dict,
    document_uuid: str,
    pipeline_run_id: str,
    ontology: dict,
    *,
    source_storage_key: str | None = None,
) -> MergedExtraction:
```

Edit DocumentEntity construction inside walk() (around the existing `doc_entity = DocumentEntity(document_number=document_number)`):

```python
    doc_entity: DocumentEntity | None = (
        DocumentEntity(
            document_number=document_number,
            storage_key=source_storage_key,
        )
        if document_number is not None
        else None
    )
```

- [ ] **Step 3g.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_docling_anchor_walker.py -v -k "storage_key"
```

Expected: 2 PASS.

- [ ] **Step 3g.5: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): source_storage_key kwarg populates DocumentEntity.storage_key

FIGURE/IMAGE.storage_key still null in this change — per-picture keys
need Artifact.self_ref migration (out of scope). Design §4.4."
```

### Task 3h: Call-site update in pipeline.py

- [ ] **Step 3h.1: Update call-site at pipeline.py:4316**

Change:

```python
        merged = _docling_anchors.walk(doc_json, document_id, run_id, ontology)
```

to:

```python
        # §4.4 — propagate Document SQL row's storage_key into the walker
        # so DocumentEntity.storage_key lands on the graph without a
        # downstream SQL join.
        document_row = db.query(Document).filter(Document.id == document_id).first()
        source_storage_key = getattr(document_row, "storage_key", None) if document_row is not None else None

        merged = _docling_anchors.walk(
            doc_json, document_id, run_id, ontology,
            source_storage_key=source_storage_key,
        )
```

Add the `Document` model import near the top of the file if it's not already in scope in that module (spot-check by grepping for `from app.models.ingest import` in pipeline.py).

- [ ] **Step 3h.2: Run integration smoke**

```bash
pytest tests/integration/test_docling_anchors_smoke.py -v
```

Expected: PASS (no regression; storage_key wiring is exercised end-to-end in §5).

- [ ] **Step 3h.3: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat(pipeline): pass source_storage_key into anchor walker

Populates DocumentEntity.storage_key from the Document SQL row so
graph→MinIO retrieval doesn't need a SQL join. Design §4.4."
```

---

## Chunk 3: Tests + VERIFICATION_CHECKLIST (Task 5)

**Blocked by:** Chunk 2.

Most unit tests landed TDD-style during Chunk 2. This chunk rounds out coverage and extends the integration smoke test.

### Task 5a: Interleaved-pictures fixture + regression test

- [ ] **Step 5a.1: Build fixture `tests/fixtures/docling_anchors/interleaved_pictures.json`**

Write a DoclingDocument JSON with at least 3 pictures in this order: one captioned "Figure 1", one uncaptioned, one captioned "Figure 2", each adjacent to body text. Use existing `with_figures_tables.json` as a template for structure.

```bash
python - <<'PY' > tests/fixtures/docling_anchors/interleaved_pictures.json
# Script that builds the fixture via DoclingDocument API. Keep the
# fixture deterministic — hand-crafted self_refs + captions. Commit
# the resulting JSON.
PY
```

- [ ] **Step 5a.2: Add regression test in `tests/integration/test_docling_anchors_smoke.py`**

```python
def test_interleaved_pictures_preserve_neighbor_alignment():
    """Regression for §4.3 zip fix: captioned + uncaptioned pictures
    interleaved in doc order must get the correct NEAR_TEXT neighbors."""
    doc_json = _load("interleaved_pictures.json")
    merged = walk(doc_json, "doc-1", "run-1", ontology={})
    # Assert picture 1 (FIGURE, captioned) has its own neighbors;
    # picture 2 (IMAGE) has its own neighbors; no cross-talk.
    ...
```

- [ ] **Step 5a.3: Run the smoke suite**

```bash
pytest tests/integration/test_docling_anchors_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5a.4: Commit**

```bash
git add tests/fixtures/docling_anchors/interleaved_pictures.json \
        tests/integration/test_docling_anchors_smoke.py
git commit -m "test(anchors): interleaved-pictures fixture + NEAR_TEXT regression"
```

### Task 5b: Integration smoke — end-to-end with new entity types

- [ ] **Step 5b.1: Extend `tests/integration/test_docling_anchors_smoke.py`**

Add one end-to-end assertion that exercises all new entity types:

```python
def test_smoke_emits_image_and_text_block_entities():
    doc_json = _load("with_figures_tables.json")
    merged = walk(doc_json, "doc-1", "run-1", ontology={}, source_storage_key="test/key.pdf")
    entity_types = {e.identity.entity_type for e in merged.entities}
    assert "DOCUMENT" in entity_types
    assert "SECTION" in entity_types
    # The with_figures_tables.json fixture has at least one uncaptioned pic
    # after §3c splits — if not, adjust the fixture or switch to
    # interleaved_pictures.json. This test should assert the expected
    # distribution given the specific fixture used.
    # Also assert section-level edges present.
    edge_types = {e.rel_type for e in merged.edges}
    assert "HAS_FIGURE" in edge_types or "HAS_IMAGE" in edge_types
```

- [ ] **Step 5b.2: Run**

```bash
pytest tests/integration/test_docling_anchors_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5b.3: Commit**

```bash
git add tests/integration/test_docling_anchors_smoke.py
git commit -m "test(anchors): smoke assertion for IMAGE + TEXT_BLOCK + SECTION edges"
```

### Task 5c: Update VERIFICATION_CHECKLIST.md

- [ ] **Step 5c.1: Read current checklist**

```bash
cat VERIFICATION_CHECKLIST.md
```

- [ ] **Step 5c.2: Append anchor-walker additions to the relevant section**

Add a new subsection (or extend the existing "Ontology / Graph Model" section) with:

```markdown
### Document-structure anchors (2026-04-21)

- [ ] New entity types: IMAGE, TEXT_BLOCK (in addition to existing DOCUMENT/SECTION/FIGURE/TABLE)
- [ ] New relationships: HAS_IMAGE (DOCUMENT→IMAGE, SECTION→IMAGE), NEAR_TEXT (FIGURE/IMAGE → TEXT_BLOCK)
- [ ] SECTION-level attribution: SECTION→FIGURE, SECTION→TABLE, SECTION→IMAGE HAS_* edges present
- [ ] DocumentEntity.storage_key populated from Document SQL row
- [ ] FigureEntity.storage_key + ImageEntity.storage_key are schema fields but emission is null until Artifact.self_ref migration lands (tracked separately)
- [ ] Re-ingestion or one-shot migration required for legacy documents whose uncaptioned pictures are currently FIGUREs — they will re-classify to IMAGE on re-ingest
```

- [ ] **Step 5c.3: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all green.

- [ ] **Step 5c.4: Commit**

```bash
git add VERIFICATION_CHECKLIST.md
git commit -m "docs(verification): document-structure anchor additions"
```

---

## Chunk 4: Notebooks (Task 6)

**Blocked by:** Chunk 3.

Both notebooks need to reflect the new entity types + relationships so they stay usable as self-documenting walkthroughs.

### Task 6a: ingest_walkthrough.ipynb

**Files:**
- Modify: `notebooks/ingest_walkthrough.ipynb`

Per the notebook-editing workflow used elsewhere in the session:
1. Close the notebook in JupyterLab (or "Reload from disk") before editing.
2. Use `NotebookEdit` (with `cell_id`) for targeted cell replacement. Fall back to a Python script that opens the JSON, mutates the targeted cell's `source`, and writes it back — when `Read` of the whole .ipynb exceeds token limits.

- [ ] **Step 6a.1: Identify the cells that list entity types / relationships**

Grep:

```bash
grep -n '"DOCUMENT"\|"SECTION"\|"FIGURE"\|"TABLE"\|"HAS_FIGURE"\|id":' notebooks/ingest_walkthrough.ipynb | head -40
```

Find the cell(s) that enumerate ontology entities/relationships for display and the cell(s) that inspect pass results post-merge.

- [ ] **Step 6a.2: Update ontology-display cell to include IMAGE + TEXT_BLOCK**

Extend any entity-list rendering to include the two new types. If the cell iterates `ALL_ENTITIES` dynamically, no code change may be needed — just re-run it.

- [ ] **Step 6a.3: Add a "Find text blocks near figures" cell**

Insert a new code cell that queries ArcadeDB for a figure and its NEAR_TEXT neighbors:

```python
# Show each FIGURE in this document and its surrounding text context.
from app.services.graph_store import get_graph_store

store = get_graph_store()
q = """
SELECT expand(out('NEAR_TEXT')) FROM FIGURE
 WHERE document_id = :doc_id
"""
# Or the equivalent ArcadeDB SQL / gremlin depending on the helper exposed.
# Render result as a small pandas DataFrame.
...
```

(Exact query depends on `graph_store` method signatures — check `app/services/arcadedb_graph.py` for an existing helper before composing raw SQL.)

- [ ] **Step 6a.4: Re-execute the notebook top-to-bottom on a real ingested document**

Start from a clean kernel. Verify every cell runs without error. Save the notebook with executed outputs.

- [ ] **Step 6a.5: Commit**

```bash
git add notebooks/ingest_walkthrough.ipynb
git commit -m "docs(nb): surface IMAGE + TEXT_BLOCK + NEAR_TEXT in ingest_walkthrough"
```

### Task 6b: raw_libraries_walkthrough.ipynb

**Files:**
- Modify: `notebooks/raw_libraries_walkthrough.ipynb`

If `raw_libraries_walkthrough.ipynb` reproduces walker logic in-line (per prior session memory, it does for some phases), mirror the new FIGURE/IMAGE split + NEAR_TEXT emission so the notebook stays byte-parity with production.

- [ ] **Step 6b.1: Identify walker-reproducing cells**

```bash
grep -n "docling_anchors\|iterate_items\|figures\|pictures" notebooks/raw_libraries_walkthrough.ipynb | head -20
```

- [ ] **Step 6b.2: Update those cells to split FIGURE vs IMAGE + emit TEXT_BLOCK + NEAR_TEXT**

If the notebook imports the production walker directly (`from app.services.docling_anchors import walk`), no code change is needed — just re-run. If it reimplements the walker inline, port the §3c / §3e changes.

- [ ] **Step 6b.3: Re-execute top-to-bottom; save with executed outputs**

- [ ] **Step 6b.4: Commit**

```bash
git add notebooks/raw_libraries_walkthrough.ipynb
git commit -m "docs(nb): update raw_libraries_walkthrough for anchor walker v2"
```

---

## Completion

After Chunk 4 commits:

- [ ] **Final step: Full-suite sanity run**

```bash
pytest tests/ -q
```

Expected: all green. No skips introduced by this work.

- [ ] **Push the branch**

```bash
git push
```

- [ ] **Announce completion**

Tell the operator (you) that:
1. Design doc at `docs/plans/2026-04-21-document-structure-pass-design.md` (v2.3).
2. Implementation plan at `docs/plans/2026-04-21-document-structure-pass-plan.md` (this file).
3. All four chunks complete; test suite green.
4. Recommend a re-ingest of at least one document with uncaptioned pictures to verify IMAGE + TEXT_BLOCK + NEAR_TEXT land in ArcadeDB end-to-end.

---

## Appendix — References

- Design: `docs/plans/2026-04-21-document-structure-pass-design.md` (v2.3)
- Existing walker: `app/services/docling_anchors.py`
- Existing walker tests: `tests/unit/test_docling_anchor_walker.py`
- Existing integration smoke: `tests/integration/test_docling_anchors_smoke.py`
- Existing fixtures: `tests/fixtures/docling_anchors/{empty_structure,sa2_minimal,with_document_number,with_figures_tables}.json`
- ALL_ENTITIES: `ontology_bundles/air_defense_v3/entities.py:1285`
- RelationshipType enum: `ontology_bundles/air_defense_v3/relationships.py:20-76`
- VALIDATION_MATRIX: `ontology_bundles/air_defense_v3/validation_matrix.py:26`
- Call-site: `app/workers/pipeline.py:4316`
