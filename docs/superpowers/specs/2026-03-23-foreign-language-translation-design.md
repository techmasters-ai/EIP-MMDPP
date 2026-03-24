# Foreign Language Detection and Translation — Design Spec

**Date:** 2026-03-24
**Status:** Approved (v2 — redesigned storage)

## Overview

Detect non-English content in document text and OCR output during ingest. When detected, translate element-by-element via LLM. Store translations in a new `translated_text` column on `DocumentElement` — `content_text` stays as the original, never overwritten. Downstream pipeline stages read `translated_text or content_text` so they always get English for embedding, metadata, and ontology extraction. The DoclingViewer shows translations as hover tooltips on the original page layout via the Docling web component's annotation system.

Image descriptions are excluded — the picture description prompt already handles foreign-language text.

## Data Model Change

**New column** on `ingest.document_elements`:
```sql
ALTER TABLE ingest.document_elements ADD COLUMN translated_text TEXT;
```

- `content_text` — always the original extracted text (never overwritten)
- `translated_text` — English translation (NULL if element is English or too short to detect)

**Alembic migration:** `0010_add_translated_text.py`

**ORM update** in `app/models/ingest.py`:
```python
translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```

## Pipeline

### `detect_and_translate` task

Same position in chain: after `prepare_document`, before the first chord.

**Updated behavior:**
1. Query `DocumentElement` rows where `content_text IS NOT NULL` and `element_type IN ('text', 'heading', 'table', 'equation')`
2. Per-element language detection (CJK threshold 5 chars, other 50 chars)
3. For non-English elements, translate via LLM in batches
4. Store translation in `translated_text` column (NOT `content_text`)
5. Set `document_metadata` keys: `detected_language`, `has_translation`
6. Upload `docling_document_translated.md` to MinIO (for the translation endpoint)

### Downstream tasks — read `translated_text or content_text`

Every task that reads `DocumentElement.content_text` for processing needs a one-line change to prefer `translated_text`:

```python
text = elem.translated_text or elem.content_text
```

**Tasks affected:**

| Task | What it reads | Change |
|---|---|---|
| `derive_document_metadata` | Loads markdown from MinIO | Try `docling_document_translated.md` first (unchanged from v1) |
| `derive_text_chunks_and_embeddings` | `DocumentElement.content_text` via `element_dicts` | Use `elem.translated_text or elem.content_text` |
| `derive_text_chunks_and_embeddings` (image desc pass) | `DocumentElement.content_text` for image elements | No change — image descriptions are already English |
| `derive_ontology_graph` | `DocumentElement.content_text` concatenated | Use `elem.translated_text or elem.content_text` |
| `derive_picture_descriptions` | `DocumentElement.content_text` for image elements | No change — image elements excluded from translation |

## API

### Existing endpoints — no changes

- `GET /v1/documents/{document_id}/metadata` — returns `detected_language` and `has_translation`
- `GET /v1/documents/{document_id}/translation` — returns assembled translated markdown from MinIO

### New endpoint for per-element translations

`GET /v1/documents/{document_id}/element-translations`

Returns element-level translations for the DoclingViewer tooltip:

```json
[
  {
    "element_uid": "abc123",
    "original_text": "参考文献",
    "translated_text": "References"
  }
]
```

Only returns elements where `translated_text IS NOT NULL`. The viewer uses `element_uid` to match translations to Docling web component elements.

## DoclingViewer — Hover Tooltips

Replace the current "Translate" mode (flat markdown dump) with a tooltip-based approach using the Docling web component's annotation system.

**When `metadata.has_translation` is true:**
1. Show "Translate" toggle button in the header
2. When toggled ON, fetch `/v1/documents/{documentId}/element-translations`
3. Inject translation annotations into the Docling JSON before passing to the web component
4. Each translated element gets an annotation: `{"kind": "translation", "text": translatedText}`
5. The `docling-img` component renders tooltips on hover for elements with annotations

**Implementation:** The existing `buildDoclingHtml` function constructs the iframe HTML with the Docling JSON. When translate mode is active, modify the JSON to add translation annotations to elements that have translations, then re-render the iframe with the modified JSON.

**Fallback:** If the Docling web component doesn't support custom `kind: "translation"` annotations natively, use the same `kind: "description"` annotation type that already works for image descriptions — it renders tooltips with hover behavior.

## Storage

| Location | Content | When |
|---|---|---|
| `document_elements.content_text` | Original extracted text | Always (never overwritten) |
| `document_elements.translated_text` | English translation | Non-English elements only |
| `docling_document_translated.md` in MinIO | Assembled translated markdown | When any translations exist |
| `document_metadata.detected_language` | ISO 639-1 code | Always (set by detect_and_translate) |
| `document_metadata.has_translation` | boolean | Always (set by detect_and_translate) |

## What Doesn't Change

- Image descriptions — prompt handles foreign text
- CLIP image embeddings — pixel-based
- `content_text` — preserved as original, never modified by translation
- Pipeline chain structure — `detect_and_translate` stays in same position

## Error Handling

- Translation failure: `translated_text` stays NULL, downstream tasks fall back to `content_text` (original language). Embedding quality degraded for that element but pipeline completes.
- `has_translation` only set `true` after at least one element has `translated_text` populated.
- Boundary marker parse failure: fall back to individual element translation.
