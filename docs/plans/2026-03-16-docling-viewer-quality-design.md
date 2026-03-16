# DoclingDocument Viewer Quality Fix — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix the DoclingDocument viewer output quality — improve VLM extraction settings, post-process markdown to remove repetition/noise, map images inline, and support text files.

**Architecture:** Three-layer fix: (1) optimize Docling VLM config for better extraction, (2) post-process markdown to deduplicate/clean noise, (3) map image placeholders to real artifact URLs and support text file fallback.

**Tech Stack:** No new dependencies. Changes to Docling converter config, new cleanup module, and backend endpoint logic.

---

## 1. Docling VLM Configuration Fix

**File:** `docker/docling/app/converter.py`

Update `init_converter()` to use modern `VlmPipelineOptions` API:

- `force_backend_text=True` — use PDF's embedded text layer instead of VLM-OCR'd text. Eliminates hallucination/repetition for text-heavy PDFs.
- `generate_picture_images=True` — extract actual image crops for inline display
- `images_scale=2.0` — 2x page resolution for better VLM accuracy
- Remove deprecated `vlm_model_name`/`vlm_model_device`/`vlm_model_dtype` params
- Use proper `vlm_options` with `max_new_tokens=8192` for complete page conversion

## 2. Markdown Post-Processing

**New file:** `docker/docling/app/cleanup.py`

Lightweight cleanup function, no external dependencies:

1. Deduplicate consecutive identical lines (3+ in a row → one)
2. Strip repetitive watermark/footer spam (lines appearing 5+ times anywhere → keep first only)
3. Remove bare `<!-- image -->` comment lines
4. Collapse excessive blank lines (max 2 consecutive)
5. Strip trailing whitespace per line

Applied in `convert_document()` — `ConvertResponse.markdown` gets cleaned version. `document_json` remains raw (available via JSON toggle).

## 3. Image Mapping in the Viewer

**File:** `app/api/v1/sources.py` — `get_docling_document` endpoint

After fetching markdown from MinIO, replace remaining `<!-- image -->` placeholders with `![image](url)` tags:

- Query `DocumentElement` rows with `element_type='image'` ordered by `element_order`
- Replace the Nth `<!-- image -->` with the Nth image artifact's proxy URL
- `react-markdown` renders them inline naturally — no frontend changes needed

## 4. Text File Support

**File:** `app/api/v1/sources.py` — `get_docling_document` endpoint

If no DoclingDocument files exist in MinIO:

- Check document `mime_type` — if `text/plain`, download original from `eip-raw` bucket
- Return raw text as `markdown` field (rendered through `react-markdown`)
- Set `document_json` to `{}`, `images` to `[]`

**File:** `frontend/src/components/DoclingViewer.tsx`

- Hide JSON toggle when `document_json` is empty

## 5. Files Summary

**Create:**
- `docker/docling/app/cleanup.py` — markdown dedup/cleanup functions

**Modify:**
- `docker/docling/app/converter.py` — VlmPipelineOptions + apply cleanup
- `app/api/v1/sources.py` — image placeholder mapping + text file fallback
- `frontend/src/components/DoclingViewer.tsx` — hide JSON toggle when empty

**No changes to:**
- Database schema / migrations
- Pipeline `prepare_document` (stores whatever converter returns)
- `DoclingDocumentResponse` schema
- Graph visualization components

**Note:** Existing documents need reingest to benefit from VLM config improvements. Image mapping and text file fallback work immediately.
