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
    "picture_descriptions": {
      "#/pictures/0": { "text": "Image shows a phased array radar...", "model": "gemma3:27b" }
    },
    "context": {
      "summary": "This document describes...",
      "classification": "UNCLASSIFIED"
    }
  }
}
```

Canonical `.text` fields on `texts[]` and `pictures[]` items are **never mutated**. They stay in the original language. Picture `meta.description` is populated natively. Translations and descriptions are overlays in `_enrichments`.

### Building an enriched copy for native tools

When HybridChunker or Docling-Graph needs the enriched document:

```python
def _build_enriched_copy(doc_dict: dict) -> dict:
    """Build a temporary copy with translations applied to .text fields
    and picture descriptions applied, for native tool consumption."""
    import copy
    enriched = copy.deepcopy(doc_dict)
    enrichments = enriched.pop("_enrichments", {})

    # Apply translations to canonical .text
    for self_ref, trans in enrichments.get("translations", {}).items():
        collection, idx = _parse_self_ref(self_ref)
        if collection in enriched and idx < len(enriched[collection]):
            enriched[collection][idx]["text"] = trans["translated_text"]

    # Apply picture descriptions to meta.description
    for self_ref, desc in enrichments.get("picture_descriptions", {}).items():
        collection, idx = _parse_self_ref(self_ref)
        if collection in enriched and idx < len(enriched[collection]):
            item = enriched[collection][idx]
            item.setdefault("meta", {})["description"] = {
                "text": desc["text"],
                "created_by": f"llm:{desc.get('model', 'unknown')}",
            }

    # Inject context as a leading text node
    context = enrichments.get("context", {})
    if context.get("summary") or context.get("classification"):
        context_text = ""
        if context.get("summary"):
            context_text += f"[Document Summary]: {context['summary']}\n"
        if context.get("classification"):
            context_text += f"[Classification]: {context['classification']}\n"
        if context_text and "texts" in enriched:
            enriched["texts"].insert(0, {
                "self_ref": "#/_context/0",
                "text": context_text,
                "label": "paragraph",
                "parent": {"$ref": "#/body"},
                "children": [],
            })

    return enriched
```

This function is called in `derive_text_chunks_and_embeddings` and `derive_ontology_graph` to build a temp copy. The persisted JSON in MinIO is never modified to contain translated text.

---

## Stage-by-stage changes

### `prepare_document`
- Extract `self_ref` from each DocItem in `_extract_elements` (converter.py)
- Build `_enrichments.identity_map` in the JSON
- Store `self_ref` in `DocumentElement.element_metadata` JSONB
- Persist JSON to MinIO (unchanged otherwise)

### `detect_and_translate`
- Load `docling_document.json` from MinIO
- For each translated element: store in `_enrichments.translations[self_ref]`
- Increment `_enrichments.version`
- Re-persist JSON to MinIO (canonical `.text` unchanged)
- Still update `DocumentElement.translated_text` (backward compat)
- Regenerate `docling_document_translated.md` from an enriched copy: build temp copy with translations applied, then `DoclingDocument.model_validate(enriched).export_to_markdown()`

### `derive_picture_descriptions`
- Load `docling_document.json` from MinIO
- For each described picture:
  - Store in `_enrichments.picture_descriptions[self_ref]`
  - Write native `meta.description` on the picture item (this IS a canonical mutation — `PictureMeta.description` is the native field for this)
  - Write legacy `annotations[kind=description]` for viewer hover contract
- Increment `_enrichments.version`
- Re-persist JSON to MinIO
- Still update `DocumentElement.content_text` (backward compat)
- Regenerate `docling_document_translated.md` from enriched copy (so it includes picture descriptions in the translated view)
- Append description appendix to `docling_document.md` (current behavior)

### `derive_text_chunks_and_embeddings`
- Load `docling_document.json` from MinIO
- Build enriched copy via `_build_enriched_copy()`
- Reconstruct `DoclingDocument.model_validate(enriched_copy)`
- Chunk with `HybridChunker(tokenizer=HuggingFaceTokenizer(...))`
- Chunk schema mapping per v4 rules
- Image description secondary pass from `_enrichments.picture_descriptions`
- Fallback: `structure_aware_chunk()` if reconstruction fails

### `derive_ontology_graph`
- Load `docling_document.json` from MinIO
- Build enriched copy via `_build_enriched_copy()` (has translations + descriptions + context)
- Pass enriched copy to Docling-Graph service
- Remove `_enriched_text` reconstruction

### Docling-Graph wrapper
- `run_extraction_pipeline` receives the enriched copy directly
- No special `_enrichments` handling needed — the copy already has translated `.text` and context node

---

## API endpoints

| Endpoint | Behavior |
|----------|----------|
| `GET /docling-raw` | Returns persisted JSON from MinIO (original language, with `_enrichments` overlay). For v4 docs: remove runtime annotation injection (descriptions are in `meta.description` + `annotations`). For pre-v4: keep injection. |
| `GET /docling` | Serves `docling_document.md` (original + picture appendix). No change. |
| `GET /translation` | Serves `docling_document_translated.md` (now regenerated from enriched DoclingDocument). No change to endpoint. |
| `GET /element-translations` | Reads from `_enrichments.translations`, reverse-maps via `identity_map`. Fallback to Postgres for pre-v4. |

### Viewer contract preserved
- Default view: `/docling-raw` returns original-language JSON → viewer displays original
- Toggle translate: frontend calls `/element-translations`, swaps `.text` fields client-side → viewer shows translated
- Hover tooltips: `annotations[kind=description]` present in persisted JSON → tooltips work

---

## Pre-v4 backward compat

| Feature | v6 path | Pre-v4 fallback |
|---------|---------|-----------------|
| `/element-translations` | `_enrichments.translations` | Postgres `DocumentElement.translated_text` |
| `/docling-raw` | Serve MinIO JSON directly | Runtime annotation injection |
| HybridChunker | Enriched copy → DoclingDocument | `structure_aware_chunk()` over DocumentElement rows |
| Graph extraction | Enriched copy to Docling-Graph | `_enriched_text` from DocumentElement rows |
| Chunk artifact resolution | `identity_map` | DocumentElement-based mapping |

Check: `doc_dict.get("_enrichments", {}).get("version")` — if absent, pre-v4.

---

## Files changed

| File | Change |
|------|--------|
| `docker/docling/app/converter.py` | Extract `self_ref` in `_extract_elements` |
| `app/services/docling_client.py` | Pass `self_ref` in metadata |
| `app/workers/pipeline.py` (prepare) | Build `_enrichments.identity_map`, store `self_ref` |
| `app/workers/pipeline.py` (translate) | Store translations in `_enrichments.translations`, regenerate translated markdown from enriched copy |
| `app/workers/pipeline.py` (pictures) | Store descriptions in `_enrichments.picture_descriptions` + native `meta.description` + legacy `annotations`, regenerate translated markdown |
| `app/workers/pipeline.py` (chunks) | Build enriched copy, HybridChunker, fallback |
| `app/workers/pipeline.py` (graph) | Build enriched copy, pass to service, remove `_enriched_text` |
| `app/workers/pipeline.py` | New `_build_enriched_copy()` helper |
| `app/api/v1/sources.py` (docling-raw) | Version-check: v4 → direct serve, pre-v4 → runtime injection |
| `app/api/v1/sources.py` (element-translations) | Read from `_enrichments`, fallback to Postgres |
| `docker/docling-graph/app/main.py` | Receive enriched copy directly (no `_enrichments` handling needed) |
| `app/services/chunking.py` | Keep as fallback |
