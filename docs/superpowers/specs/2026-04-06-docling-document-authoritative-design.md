# Design: DoclingDocument as Authoritative Mutable Artifact (v6)

**Date:** 2026-04-06
**Status:** Final — all review findings resolved
**TODO items:** #60, #61, #62

---

## Core Principle

The persisted `docling_document.json` in MinIO is the **authoritative artifact** in **original language**. Translations and picture descriptions are stored as **enrichment overlays** within the JSON, NOT as mutations to canonical `.text` fields. Native tools (HybridChunker, Docling-Graph) receive a **temporary enriched copy** built at call time.

This preserves:
- The viewer's default display (original language JSON)
- The translation toggle (overlay swap in frontend)
- The `/docling-raw` endpoint contract (serves original-language JSON)
- Native tool input quality (enriched copy has translated text + descriptions)

---

## Enrichment Storage Model

```json
{
  "texts": [...],
  "pictures": [...],
  "_enrichments": {
    "version": 3,
    "identity_map": { "#/texts/0": "1-0-text-a1b2c3", ... },
    "translations": {
      "#/texts/0": { "original_text": "Зенитный...", "translated_text": "Anti-aircraft...", "language": "ru" },
      "#/texts/1": { "original_text": "Комплекс...", "translated_text": "System...", "language": "ru" }
    },
    "context": {
      "summary": "This document describes...",
      "classification": "UNCLASSIFIED"
    }
  }
}
```

Canonical `.text` fields on `texts[]` and `pictures[]` items are **never mutated** with translations. They stay in the original language. Picture `meta.description` IS populated natively (it is the canonical Docling field for this purpose — not a mutation of `.text`).

### Picture description precedence rule

`meta.description` is **canonical**. `annotations[kind=description]` is derived for viewer compatibility. `_enrichments.picture_descriptions` is removed — unnecessary third copy. The pipeline writes `meta.description` on the picture item and derives the legacy annotation from it. Reads always check `meta.description` first.

### Two enrichment copy functions (chunking vs graph extraction)

Chunking and graph extraction need different enriched copies. Chunking should NOT see the synthetic context node (which would create a spurious chunk and break artifact resolution). Graph extraction DOES need the context.

```python
def _build_enriched_copy_for_chunking(doc_dict: dict) -> dict:
    """Enriched copy for HybridChunker: translations applied, no context node."""
    import copy
    enriched = copy.deepcopy(doc_dict)
    enrichments = enriched.pop("_enrichments", {})

    for self_ref, trans in enrichments.get("translations", {}).items():
        collection, idx = _parse_self_ref(self_ref)
        if collection not in enriched or idx >= len(enriched[collection]):
            continue
        item = enriched[collection][idx]

        if collection == "tables":
            # Tables are structured: content lives in data.table_cells[].text,
            # not a top-level .text field. The pipeline translates tables as
            # monolithic text blocks, so we can't map translated text back to
            # individual cells. Instead, replace the table's data with a
            # single-cell table containing the full translated text. This
            # passes model_validate(), is chunked by HybridChunker, and is
            # seen by Docling-Graph.
            item["data"] = {
                "table_cells": [{
                    "text": trans["translated_text"],
                    "row_span": 1, "col_span": 1,
                    "start_row_offset_idx": 0, "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0, "end_col_offset_idx": 1,
                    "column_header": False, "row_header": False,
                    "row_section": False, "fillable": False,
                }],
                "num_rows": 1, "num_cols": 1,
            }
        else:
            # TextItem, FormulaItem, SectionHeaderItem — all have .text
            item["text"] = trans["translated_text"]

    # meta.description is already on picture items (written during
    # derive_picture_descriptions), no overlay needed

    return enriched


def _build_enriched_copy_for_graph(doc_dict: dict) -> dict:
    """Enriched copy for Docling-Graph: translations + context node.
    Uses native DoclingDocument.add_text() for safe tree insertion."""
    from docling.datamodel.document import DoclingDocument

    enriched = _build_enriched_copy_for_chunking(doc_dict)

    # Inject context via native API (not raw dict surgery)
    context = doc_dict.get("_enrichments", {}).get("context", {})
    if context.get("summary") or context.get("classification"):
        doc = DoclingDocument.model_validate(enriched)
        context_text = ""
        if context.get("summary"):
            context_text += f"[Document Summary]: {context['summary']}\n"
        if context.get("classification"):
            context_text += f"[Classification]: {context['classification']}\n"
        if context_text:
            doc.add_text(text=context_text, label="paragraph")
        enriched = doc.export_to_dict()

    return enriched
```

The context node is added via `DoclingDocument.add_text()` which handles `orig`, `parent`, `self_ref`, and body tree insertion correctly. It appends to the end of the body, which is acceptable — graph extraction does not depend on context position.

### Translated markdown with picture descriptions

Native `export_to_markdown()` does NOT serialize `PictureMeta.description` as markdown text. To include picture descriptions in translated markdown:

1. Call `export_to_markdown()` on the enriched copy (has translated `.text`)
2. Append a picture-description appendix (same format as current `derive_picture_descriptions` appendix)
3. The appendix is built from `meta.description` on picture items, not from `_enrichments`

```python
def _regenerate_translated_markdown(doc_dict: dict) -> str:
    """Generate translated markdown with picture description appendix.

    Tables: the enriched copy replaces table data with a single-cell
    table containing the translated text, so export_to_markdown() renders
    the translated content natively (no post-processing needed).
    """
    enriched = _build_enriched_copy_for_chunking(doc_dict)
    doc = DoclingDocument.model_validate(enriched)
    md = doc.export_to_markdown()

    # Append picture descriptions (not included by native export)
    pic_descs = []
    for pic in doc_dict.get("pictures", []):
        meta = pic.get("meta") or {}
        desc = meta.get("description", {})
        if desc.get("text"):
            pic_descs.append(f"**Image:** {desc['text']}")
    if pic_descs:
        md += "\n\n---\n## Image Descriptions\n\n" + "\n\n".join(pic_descs)

    return md
```

**Table translation strategy:** The pipeline translates tables as monolithic text blocks (the LLM receives the full table text and returns a translated version). Since we can't reliably split translated text back into individual cells, the enriched copy replaces the table's `TableData` with a single-cell table containing the full translated text. This approach: (a) passes `DoclingDocument.model_validate()`, (b) is chunked correctly by HybridChunker, (c) is consumed by Docling-Graph, and (d) renders as translated content via `export_to_markdown()`. The trade-off is that table structure (rows/columns) is lost in the translated enriched copy — the original structure is preserved in the persisted JSON for the viewer.

This is called in both `detect_and_translate` (after storing translations) and `derive_picture_descriptions` (after storing descriptions) so translated markdown stays in sync.

---

## Stage-by-stage changes

### `prepare_document`
- Extract `self_ref` from each DocItem in `_extract_elements` (converter.py)
- Build `_enrichments.identity_map` in the JSON
- **Initialize `_enrichments.version = 0`** so ALL new documents are enriched-path from the start (even if translation and picture-description stages are skipped for all-English, no-image documents)
- Store `self_ref` in `DocumentElement.element_metadata` JSONB
- Persist JSON to MinIO

### `detect_and_translate`
- Load `docling_document.json` from MinIO
- For each translated element: store in `_enrichments.translations[self_ref]`
- Increment `_enrichments.version`
- Re-persist JSON to MinIO (canonical `.text` unchanged)
- Still update `DocumentElement.translated_text` (backward compat)
- Regenerate `docling_document_translated.md` via `_regenerate_translated_markdown()` (enriched copy + picture description appendix)

### `derive_picture_descriptions`
- Load `docling_document.json` from MinIO
- For each described picture:
  - Write native `meta.description` on the picture item (canonical — `PictureMeta.description`)
  - Derive legacy `annotations[kind=description]` from `meta.description` for viewer hover
  - (No `_enrichments.picture_descriptions` — `meta.description` is the single source)
- Increment `_enrichments.version`
- Re-persist JSON to MinIO
- Still update `DocumentElement.content_text` (backward compat)
- Regenerate `docling_document_translated.md` via `_regenerate_translated_markdown()` (includes picture descriptions in appendix)
- Append description appendix to `docling_document.md` (current behavior)

### `derive_text_chunks_and_embeddings`
- Load `docling_document.json` from MinIO
- Build enriched copy via `_build_enriched_copy_for_chunking()` (translations applied, NO context node)
- Reconstruct `DoclingDocument.model_validate(enriched_copy)`
- Chunk with `HybridChunker(tokenizer=HuggingFaceTokenizer(...))`
- Chunk schema mapping per earlier rules
- Image description secondary pass from picture items' `meta.description`
- Fallback: `structure_aware_chunk()` if reconstruction fails

### `derive_ontology_graph`
- Load `docling_document.json` from MinIO
- Build enriched copy via `_build_enriched_copy_for_graph()` (translations + descriptions + context node via native `add_text()`)
- Pass enriched copy to Docling-Graph service
- Remove `_enriched_text` reconstruction

### Docling-Graph wrapper
- `run_extraction_pipeline` receives the enriched copy directly
- No special `_enrichments` handling needed — the copy already has translated `.text` and context node added via native API

---

## API endpoints

| Endpoint | Behavior |
|----------|----------|
| `GET /docling-raw` | Returns persisted JSON from MinIO (original language, with `_enrichments` overlay). **Enriched docs** (have `_enrichments.version`): remove runtime annotation injection (descriptions are in `meta.description` + `annotations`). **Legacy docs** (no `_enrichments.version`): keep runtime injection. |
| `GET /docling` | Serves `docling_document.md` (original + picture appendix). No change. |
| `GET /translation` | Serves `docling_document_translated.md` (now regenerated from enriched DoclingDocument). No change to endpoint. |
| `GET /element-translations` | Reads from `_enrichments.translations`, reverse-maps via `identity_map`. Fallback to Postgres for legacy docs. |

### Viewer contract preserved
- Default view: `/docling-raw` returns original-language JSON → viewer displays original
- Toggle translate: frontend calls `/element-translations`, swaps `.text` fields client-side → viewer shows translated
- Hover tooltips: `annotations[kind=description]` present in persisted JSON → tooltips work

---

## Backward compatibility for legacy documents

Documents ingested before this refactor lack `_enrichments`. The runtime gate is: `doc_dict.get("_enrichments", {}).get("version")`. If the key is absent, all code paths fall back to the current behavior.

| Feature | Enriched path (has `_enrichments.version`) | Legacy fallback (no `_enrichments.version`) |
|---------|---------------------------------------------|---------------------------------------------|
| `/element-translations` | `_enrichments.translations` + `identity_map` | Postgres `DocumentElement.translated_text` |
| `/docling-raw` | Serve MinIO JSON directly (no injection) | Runtime annotation injection from Postgres |
| HybridChunker | Enriched copy → `DoclingDocument` | `structure_aware_chunk()` over DocumentElement rows |
| Graph extraction | Enriched copy → Docling-Graph service | `_enriched_text` from DocumentElement rows |
| Chunk artifact resolution | `identity_map` → `element_uid` → Artifact | DocumentElement-based mapping |

---

## Assumptions to verify

1. **Docling-Graph picture description consumption:** The wrapper passes the enriched DoclingDocument (with `meta.description` on pictures) to `run_pipeline()`. Whether Docling-Graph extracts entities from picture descriptions depends on the upstream library, not our wrapper. **This is an assumption — add an integration test** proving that enriched pictures influence extraction before removing the current fallback.

2. **Context node append vs prepend:** Native `DoclingDocument.add_text()` appends to the body. Current behavior prepends summary/classification to the text stream. Appending places context at the end of the document, not the beginning. **This is an acceptable tradeoff** because: (a) LLM extraction is not order-dependent for document-level context, (b) the alternative (raw dict manipulation to prepend) bypasses native API safety, (c) if order matters, a future improvement can use `insert_text()` or `insert_item_before_sibling()` which the library also provides.

---

## Files changed

| File | Change |
|------|--------|
| `docker/docling/app/converter.py` | Extract `self_ref` in `_extract_elements` |
| `app/services/docling_client.py` | Pass `self_ref` in metadata |
| `app/workers/pipeline.py` (prepare) | Build `_enrichments.identity_map`, store `self_ref` |
| `app/workers/pipeline.py` (translate) | Store translations in `_enrichments.translations`, regenerate translated markdown via `_regenerate_translated_markdown()` |
| `app/workers/pipeline.py` (pictures) | Write native `meta.description` + derive legacy `annotations` (NO `_enrichments.picture_descriptions`), regenerate translated markdown |
| `app/workers/pipeline.py` (chunks) | `_build_enriched_copy_for_chunking()`, HybridChunker, fallback |
| `app/workers/pipeline.py` (graph) | `_build_enriched_copy_for_graph()`, pass to service, remove `_enriched_text` |
| `app/workers/pipeline.py` | New `_build_enriched_copy_for_chunking()`, `_build_enriched_copy_for_graph()`, `_regenerate_translated_markdown()`, `_parse_self_ref()` helpers |
| `app/api/v1/sources.py` (docling-raw) | Version-check: enriched → direct serve, legacy → runtime injection |
| `app/api/v1/sources.py` (element-translations) | Read from `_enrichments`, fallback to Postgres for legacy |
| `docker/docling-graph/app/main.py` | Receive enriched copy directly (no `_enrichments` handling needed) |
| `app/services/chunking.py` | Keep as fallback |
