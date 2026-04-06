# DoclingDocument as Authoritative Mutable Artifact — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DoclingDocument JSON in MinIO the authoritative enriched artifact through the ingest pipeline, with native Docling chunking and proper enrichment overlays.

**Architecture:** Persisted JSON stays original-language. Translations and descriptions are stored as enrichment overlays (`_enrichments`). Native tools receive temporary enriched copies built at call time. HybridChunker replaces custom chunking.

**Tech Stack:** Docling (`DoclingDocument`, `HybridChunker`, `HuggingFaceTokenizer`), MinIO, PostgreSQL, Celery

**Spec:** `docs/superpowers/specs/2026-04-06-docling-document-authoritative-design.md` (v10)

---

## Task 0: Extract `self_ref` from Docling items in converter

**Files:**
- Modify: `docker/docling/app/converter.py` (~L289-430)
- Modify: `docker/docling/app/converter.py` `ConvertedElement` model

- [ ] **Step 1:** Add `self_ref` field to `ConvertedElement` (the Pydantic model returned by the Docling service). Find the model definition and add `self_ref: str | None = None`.

- [ ] **Step 2:** In `_extract_elements()`, capture `item.self_ref` for each DocItem and pass it to `ConvertedElement`. Every item in the `for item, level in doc.iterate_items()` loop has a `self_ref` attribute (e.g., `#/texts/0`).

- [ ] **Step 3:** Verify locally: `docker compose build docling` then check that `/convert` response includes `self_ref` in each element.

- [ ] **Step 4:** Commit.

---

## Task 1: Pass `self_ref` through the client into pipeline

**Files:**
- Modify: `app/services/docling_client.py` `ExtractedChunk` / mapping logic
- Modify: `app/workers/pipeline.py` `prepare_document` (~L815-935)

- [ ] **Step 1:** In `docling_client.py`, ensure `self_ref` from `ConvertedElement` is passed through to `ExtractedChunk.metadata` (or as a direct field).

- [ ] **Step 2:** In `prepare_document`, when building DocumentElement rows, store `self_ref` in `element_metadata` JSONB: `element_metadata = {**existing_meta, "self_ref": chunk.metadata.get("self_ref")}`.

- [ ] **Step 3:** Build `_enrichments` in the DoclingDocument JSON before persisting to MinIO:
```python
identity_map = {}
for elem in elements:
    self_ref = elem.metadata.get("self_ref")
    if self_ref:
        identity_map[self_ref] = elem.element_uid
doc_dict["_enrichments"] = {
    "version": 0,
    "identity_map": identity_map,
    "translations": {},
    "context": {},
}
upload_bytes_sync(json.dumps(doc_dict, ...).encode(), bucket, key)
```

- [ ] **Step 4:** Run unit tests: `.venv/bin/python -m pytest tests/unit/ -q`. Confirm 665+ pass.

- [ ] **Step 5:** Commit.

---

## Task 2: Add enrichment helper functions

**Files:**
- Create: `app/services/docling_enrichment.py`
- Test: `tests/unit/test_docling_enrichment.py`

- [ ] **Step 1:** Create `app/services/docling_enrichment.py` with these helpers:

```python
def _parse_self_ref(self_ref: str) -> tuple[str, int]:
    """Parse '#/texts/0' → ('texts', 0)."""

def _build_enriched_copy_for_chunking(doc_dict: dict) -> dict:
    """Translations applied to .text, tables replaced with single-cell translated.
    No context node. Used by HybridChunker."""

def _build_enriched_copy_for_graph(doc_dict: dict) -> dict:
    """Same as chunking + context node via DoclingDocument.add_text().
    Used by Docling-Graph."""

def _regenerate_translated_markdown(doc_dict: dict) -> str:
    """export_to_markdown() on enriched copy + picture description appendix."""
```

- [ ] **Step 2:** Write tests for each helper in `tests/unit/test_docling_enrichment.py`:
  - `_parse_self_ref` parses texts, tables, pictures
  - `_build_enriched_copy_for_chunking` applies translations, replaces table data
  - `_build_enriched_copy_for_graph` adds context node
  - `_regenerate_translated_markdown` includes picture descriptions in appendix

- [ ] **Step 3:** Run tests, confirm pass.

- [ ] **Step 4:** Commit.

---

## Task 3: Update `detect_and_translate` to enrich DoclingDocument JSON

**Files:**
- Modify: `app/workers/pipeline.py` `detect_and_translate` (~L1170-1310)

- [ ] **Step 1:** After translating elements and updating DocumentElement rows, reload the DoclingDocument JSON from MinIO.

- [ ] **Step 2:** For each translated element, look up its `self_ref` via `element_metadata` and store in `_enrichments.translations`:
```python
doc_dict["_enrichments"]["translations"][self_ref] = {
    "original_text": original,
    "translated_text": translated,
    "language": detected_lang,
}
```

- [ ] **Step 3:** Increment `_enrichments.version`.

- [ ] **Step 4:** Re-persist JSON to MinIO.

- [ ] **Step 5:** Replace the current custom markdown concatenation for `docling_document_translated.md` with `_regenerate_translated_markdown(doc_dict)`.

- [ ] **Step 6:** Run tests, commit.

---

## Task 4: Update `derive_picture_descriptions` to enrich DoclingDocument JSON

**Files:**
- Modify: `app/workers/pipeline.py` `derive_picture_descriptions` (~L1368-1545)

- [ ] **Step 1:** After generating descriptions and updating DocumentElement rows, reload the DoclingDocument JSON from MinIO.

- [ ] **Step 2:** For each described picture, write native `meta.description`:
```python
pic_item = doc_dict["pictures"][pic_idx]
pic_item.setdefault("meta", {})["description"] = {
    "text": description,
    "created_by": f"llm:{model_name}",
}
```

- [ ] **Step 3:** Derive legacy `annotations` for viewer hover:
```python
if "annotations" not in pic_item:
    pic_item["annotations"] = []
pic_item["annotations"].append({
    "kind": "description",
    "text": description,
    "source": "llm",
    "model": model_name,
})
```

- [ ] **Step 4:** Increment `_enrichments.version`, re-persist JSON.

- [ ] **Step 5:** Regenerate `docling_document_translated.md` via `_regenerate_translated_markdown()` (so translated view includes picture descriptions).

- [ ] **Step 6:** Run tests, commit.

---

## Task 5: Replace custom chunking with native HybridChunker

**Files:**
- Modify: `app/workers/pipeline.py` `derive_text_chunks_and_embeddings` (~L1620-1948)
- Keep: `app/services/chunking.py` as fallback

- [ ] **Step 1:** At the start of the chunking section, load DoclingDocument JSON from MinIO and try to reconstruct:
```python
from app.services.docling_enrichment import _build_enriched_copy_for_chunking
doc_dict = json.loads(download_bytes_sync(bucket, key))
enriched = _build_enriched_copy_for_chunking(doc_dict)
doc = DoclingDocument.model_validate(enriched)
```

- [ ] **Step 2:** If reconstruction succeeds, chunk with HybridChunker:
```python
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

tok = AutoTokenizer.from_pretrained(settings.text_embedding_model)
hf_tok = HuggingFaceTokenizer(tokenizer=tok, max_tokens=settings.chunk_max_tokens)
chunker = HybridChunker(tokenizer=hf_tok)
native_chunks = list(chunker.chunk(doc))
```

- [ ] **Step 3:** Map native chunks to TextChunk records using the chunk schema mapping from the spec (chunk_id, artifact_id, page_number, modality).

- [ ] **Step 4:** If reconstruction fails, fall back to current `structure_aware_chunk()` path.

- [ ] **Step 5:** Preserve the image-description secondary pass: scan `pictures` for `meta.description`, create image_description TextChunks.

- [ ] **Step 6:** Run full test suite, commit.

---

## Task 6: Update `derive_ontology_graph` to use enriched copy

**Files:**
- Modify: `app/workers/pipeline.py` `derive_ontology_graph` (~L2218-2300)

- [ ] **Step 1:** Replace the current enriched-text reconstruction with:
```python
from app.services.docling_enrichment import _build_enriched_copy_for_graph
doc_dict = json.loads(download_bytes_sync(bucket, key))
enriched = _build_enriched_copy_for_graph(doc_dict)
```

- [ ] **Step 2:** Add `_enrichments.context` with summary/classification to the JSON before building the enriched copy (loaded from document_metadata).

- [ ] **Step 3:** Pass `enriched` directly to `extract_graph_all()`.

- [ ] **Step 4:** Keep the fallback: if JSON is missing, reconstruct from DocumentElement rows.

- [ ] **Step 5:** Run tests, commit.

---

## Task 7: Update API endpoints for enriched JSON

**Files:**
- Modify: `app/api/v1/sources.py` `get_docling_raw` (~L818)
- Modify: `app/api/v1/sources.py` `get_element_translations` (~L916)

- [ ] **Step 1:** In `get_docling_raw`: check `_enrichments.version` in the JSON. If present (enriched doc), serve directly without runtime annotation injection. If absent (legacy), keep current injection.

- [ ] **Step 2:** In `get_element_translations`: try reading from `_enrichments.translations` + `identity_map` first. If not present, fall back to Postgres DocumentElement.translated_text.

- [ ] **Step 3:** Run tests, commit.

---

## Task 8: Update TODO.md and run final verification

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1:** Mark #60, #61, #62 as done in TODO.md.

- [ ] **Step 2:** Run full test suite: `.venv/bin/python -m pytest tests/unit/ -q`.

- [ ] **Step 3:** Final commit and push.
