# Foreign Language Translation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect non-English content per-element during ingest, translate via LLM, and provide a translation toggle in the DoclingViewer.

**Architecture:** New `detect_and_translate` Celery task inserted after `prepare_document`. Uses `langdetect` per-element, translates non-English elements via LLM in batches, updates `DocumentElement.content_text` so all downstream tasks get English. Stores translated markdown in MinIO for the viewer. New API endpoint serves the translation. DoclingViewer gets a "Translate" toggle.

**Tech Stack:** Python 3.11, langdetect, Celery, MinIO (boto3), SQLAlchemy, FastAPI, React/TypeScript

**Spec:** `docs/superpowers/specs/2026-03-23-foreign-language-translation-design.md`

---

## Chunk 1: Backend — Config + Translation Service

### Task 1: Add translation settings to config

**Files:**
- Modify: `app/config.py:141` (after graphrag settings)
- Modify: `env.example` (add new env vars)

- [ ] **Step 1: Add settings to `app/config.py`**

After the graphrag settings block (~line 141), add:

```python
    # Translation (foreign language detection + LLM translation)
    translation_enabled: bool = True
    translation_model: str = "gpt-oss:120b"
    translation_timeout: int = 300
    translation_prompt: str = "Translate the following text to English. If the text is already in English, return it unchanged. Preserve all markdown formatting including headings (#), bullet points, tables, and code blocks. Preserve technical designators, model numbers, NATO reporting names, and military identifiers verbatim — do not transliterate or translate them (e.g., keep С-75, ЗРК, 9М38 as-is). Preserve all numbers, units, and acronyms. Preserve ---ELEMENT_BOUNDARY--- markers exactly as they appear. Return only the translated text with no commentary."
    translation_soft_time_limit: int = 3600
    translation_time_limit: int = 3660
```

- [ ] **Step 2: Add env vars to `env.example`**

Add a new section after the Picture Description block:

```
# ---------------------------------------------------------------------------
# Translation (foreign language detection + LLM translation)
# ---------------------------------------------------------------------------
TRANSLATION_ENABLED=true
TRANSLATION_MODEL=gpt-oss:120b
TRANSLATION_TIMEOUT=300
TRANSLATION_SOFT_TIME_LIMIT=3600
TRANSLATION_TIME_LIMIT=3660
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/config.py env.example
git commit -m "feat: add translation configuration settings"
```

---

### Task 2: Create translation service module

**Files:**
- Create: `app/services/translation.py`
- Test: `tests/unit/test_translation.py`

This module handles language detection and LLM translation. Two public functions: `detect_element_languages()` and `translate_elements()`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_translation.py

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestDetectElementLanguages:
    def test_english_text_detected(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "This is a long enough English sentence for reliable language detection by the library.", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["document_language"] == "en"
        assert result["non_english_indices"] == []

    def test_russian_text_detected(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "Зенитная ракетная система С-75 Двина предназначена для поражения воздушных целей на средних и больших высотах.", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["document_language"] == "ru"
        assert result["non_english_indices"] == [0]

    def test_short_text_skipped(self):
        from app.services.translation import detect_element_languages
        elements = [{"content_text": "Short", "element_type": "text"}]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == []

    def test_mixed_language_document(self):
        from app.services.translation import detect_element_languages
        elements = [
            {"content_text": "This is English text that is long enough for detection by the langdetect library.", "element_type": "text"},
            {"content_text": "Зенитная ракетная система С-75 Двина предназначена для поражения воздушных целей на средних и больших высотах.", "element_type": "text"},
            {"content_text": "Another English paragraph that should be detected as English by the library.", "element_type": "text"},
        ]
        result = detect_element_languages(elements)
        assert result["non_english_indices"] == [1]
        assert result["document_language"] == "ru"

    def test_empty_elements(self):
        from app.services.translation import detect_element_languages
        result = detect_element_languages([])
        assert result["document_language"] == "en"
        assert result["non_english_indices"] == []


class TestTranslateElements:
    @patch("app.services.translation._ollama_translate")
    def test_translates_non_english_elements(self, mock_translate):
        from app.services.translation import translate_elements
        mock_translate.return_value = "Translated text"
        elements = [
            {"content_text": "Оригинальный текст", "element_order": 0},
            {"content_text": "English text", "element_order": 1},
        ]
        non_english_indices = [0]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Translated text"
        assert translated[1] == "English text"  # untouched
        mock_translate.assert_called_once()

    @patch("app.services.translation._ollama_translate")
    def test_batches_elements(self, mock_translate):
        from app.services.translation import translate_elements
        # Two short elements should be batched together
        mock_translate.return_value = "Trans A\n---ELEMENT_BOUNDARY---\nTrans B"
        elements = [
            {"content_text": "Текст А", "element_order": 0},
            {"content_text": "Текст Б", "element_order": 1},
        ]
        non_english_indices = [0, 1]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Trans A"
        assert translated[1] == "Trans B"

    @patch("app.services.translation._ollama_translate")
    def test_fallback_on_boundary_parse_failure(self, mock_translate):
        from app.services.translation import translate_elements
        # First call returns without boundaries (batch fails), then individual calls succeed
        mock_translate.side_effect = ["No boundaries here", "Trans A", "Trans B"]
        elements = [
            {"content_text": "Текст А", "element_order": 0},
            {"content_text": "Текст Б", "element_order": 1},
        ]
        non_english_indices = [0, 1]
        translated = translate_elements(elements, non_english_indices)
        assert translated[0] == "Trans A"
        assert translated[1] == "Trans B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_translation.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `app/services/translation.py`**

```python
"""Foreign language detection and LLM-based translation.

Detects non-English content per-element using langdetect, then translates
flagged elements via Ollama in batches with boundary markers.
"""

import logging
from collections import Counter

import httpx
from langdetect import detect_langs, LangDetectException
from langdetect.detector_factory import DetectorFactory

from app.config import get_settings

logger = logging.getLogger(__name__)

# Deterministic detection
DetectorFactory.seed = 0

_MIN_DETECT_LENGTH = 50
_BATCH_CHAR_LIMIT = 2000
_BOUNDARY = "\n---ELEMENT_BOUNDARY---\n"


def detect_element_languages(elements: list[dict]) -> dict:
    """Detect language per element.

    Args:
        elements: List of dicts with 'content_text' and 'element_type' keys.

    Returns:
        {
            "document_language": "ru",  # most common non-English, or "en"
            "non_english_indices": [0, 2, 5],  # indices needing translation
        }
    """
    non_english: list[int] = []
    lang_counts: Counter = Counter()

    for i, elem in enumerate(elements):
        text = elem.get("content_text", "") or ""
        if len(text) < _MIN_DETECT_LENGTH:
            continue
        try:
            langs = detect_langs(text)
            top = langs[0]
            if top.lang != "en" and top.prob > 0.7:
                non_english.append(i)
                lang_counts[top.lang] += 1
        except LangDetectException:
            continue

    doc_lang = lang_counts.most_common(1)[0][0] if lang_counts else "en"
    return {
        "document_language": doc_lang,
        "non_english_indices": non_english,
    }


def translate_elements(
    elements: list[dict],
    non_english_indices: list[int],
) -> list[str]:
    """Translate non-English elements, return all element texts (translated + untouched).

    Returns list of strings in the same order as input elements.
    """
    settings = get_settings()
    result = [elem["content_text"] for elem in elements]

    if not non_english_indices:
        return result

    # Build batches of non-English elements
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_len = 0

    for idx in non_english_indices:
        text = elements[idx]["content_text"]
        text_len = len(text)

        if text_len > _BATCH_CHAR_LIMIT:
            # Oversized element gets its own batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_len = 0
            batches.append([idx])
        elif current_len + text_len > _BATCH_CHAR_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_len = text_len
        else:
            current_batch.append(idx)
            current_len += text_len

    if current_batch:
        batches.append(current_batch)

    # Translate each batch
    for batch_indices in batches:
        if len(batch_indices) == 1:
            # Single element — no boundary markers needed
            idx = batch_indices[0]
            translated = _ollama_translate(elements[idx]["content_text"])
            if translated:
                result[idx] = translated.strip()
        else:
            # Multi-element batch with boundary markers
            combined = _BOUNDARY.join(elements[idx]["content_text"] for idx in batch_indices)
            translated = _ollama_translate(combined)

            if translated and _BOUNDARY.strip() in translated:
                parts = translated.split(_BOUNDARY.strip())
                if len(parts) == len(batch_indices):
                    for idx, part in zip(batch_indices, parts):
                        result[idx] = part.strip()
                else:
                    # Boundary count mismatch — fall back to individual translation
                    _translate_individually(elements, batch_indices, result)
            else:
                # No boundaries in response — fall back to individual translation
                _translate_individually(elements, batch_indices, result)

    return result


def _translate_individually(
    elements: list[dict], indices: list[int], result: list[str]
) -> None:
    """Fallback: translate each element individually."""
    for idx in indices:
        translated = _ollama_translate(elements[idx]["content_text"])
        if translated:
            result[idx] = translated.strip()


def _ollama_translate(text: str) -> str | None:
    """Send text to Ollama for translation."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/v1/chat/completions"
    prompt = settings.translation_prompt.replace("\\n", "\n")

    try:
        with httpx.Client(timeout=settings.translation_timeout) as client:
            resp = client.post(url, json={
                "model": settings.translation_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.1,
                "max_tokens": settings.llm_max_tokens,
            })
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return None
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_translation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/translation.py tests/unit/test_translation.py
git commit -m "feat: add translation service with per-element language detection"
```

---

## Chunk 2: Pipeline Task + Downstream Changes

### Task 3: Create `detect_and_translate` pipeline task

**Files:**
- Modify: `app/workers/pipeline.py` (add new task, update chain at line 394, update REQUIRED_STAGES at line 2645)

- [ ] **Step 1: Add the `detect_and_translate` task to `pipeline.py`**

Add the new task after the `derive_document_metadata` task definition (~after line 1100). Follow the same pattern as other tasks (bind, retries, stage_run tracking).

```python
# ---------------------------------------------------------------------------
# Stage: detect_and_translate — language detection + LLM translation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.detect_and_translate",
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=settings.translation_soft_time_limit,
    time_limit=settings.translation_time_limit,
    queue="ingest",
)
def detect_and_translate(self, document_id: str, run_id: str | None = None) -> dict:
    """Detect non-English content per-element and translate via LLM."""
    import json as json_mod

    logger.info("detect_and_translate: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="detect_and_translate")

    if not settings.translation_enabled:
        logger.info("detect_and_translate: disabled, skipping for %s", document_id)
        db = _get_db()
        try:
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "disabled"})
                db.commit()
        finally:
            db.close()
        return {"stage": "detect_and_translate", "status": "skipped"}

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "detect_and_translate", "RUNNING",
                              attempt=self.request.retries + 1)
            db.commit()

        from app.models.ingest import DocumentElement
        from app.services.translation import detect_element_languages, translate_elements
        from app.services.storage import download_bytes_sync, upload_bytes_sync
        from sqlalchemy import select

        # Load elements
        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.content_text.isnot(None),
                DocumentElement.element_type.in_(["text", "heading", "table", "equation"]),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        if not elements:
            logger.info("detect_and_translate: no text elements for %s", document_id)
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_elements"})
                db.commit()
            return {"stage": "detect_and_translate", "status": "skipped"}

        # Convert to dicts for detection
        elem_dicts = [{"content_text": e.content_text, "element_type": e.element_type} for e in elements]

        # Detect languages
        detection = detect_element_languages(elem_dicts)
        doc_lang = detection["document_language"]
        non_english = detection["non_english_indices"]

        # Update document metadata with detected language
        from app.models.ingest import Document
        doc = db.get(Document, uuid.UUID(document_id))
        meta = doc.document_metadata or {} if doc else {}
        meta["detected_language"] = doc_lang
        meta["has_translation"] = False

        if not non_english:
            # All English — skip translation
            if doc:
                doc.document_metadata = meta
                db.commit()
            if run_id:
                _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"detected_language": doc_lang, "total_elements": len(elements),
                                           "non_english_elements": 0, "skipped": True})
                db.commit()
            logger.info("detect_and_translate: all English for %s", document_id)
            return {"stage": "detect_and_translate", "status": "skipped", "language": "en"}

        # Translate non-English elements
        translated_texts = translate_elements(elem_dicts, non_english)

        # Update DocumentElement rows with translated text
        for i, elem in enumerate(elements):
            if i in non_english and translated_texts[i] != elem.content_text:
                elem.content_text = translated_texts[i]
        db.commit()

        # Build and upload translated markdown
        all_texts = [translated_texts[i] for i in range(len(elements))]
        translated_md = "\n\n".join(all_texts)
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        upload_bytes_sync(
            translated_md.encode("utf-8"),
            bucket,
            f"{base_key}/docling_document_translated.md",
            content_type="text/markdown; charset=utf-8",
        )

        meta["has_translation"] = True
        if doc:
            doc.document_metadata = meta
            db.commit()

        if run_id:
            _update_stage_run(db, run_id, "detect_and_translate", "COMPLETE",
                              attempt=self.request.retries + 1,
                              metrics={
                                  "detected_language": doc_lang,
                                  "total_elements": len(elements),
                                  "non_english_elements": len(non_english),
                                  "elements_translated": len(non_english),
                              })
            db.commit()

        logger.info("detect_and_translate: document_id=%s lang=%s translated=%d/%d",
                     document_id, doc_lang, len(non_english), len(elements))
        return {"stage": "detect_and_translate", "status": "ok", "language": doc_lang,
                "translated": len(non_english)}

    except CeleryRetry:
        raise
    except SoftTimeLimitExceeded:
        logger.warning("detect_and_translate: soft time limit for %s", document_id)
        if run_id:
            try:
                _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                                  attempt=self.request.retries + 1, error_message="soft time limit")
                db.commit()
            except Exception:
                pass
        return {"stage": "detect_and_translate", "status": "timeout"}
    except Exception as exc:
        logger.error("detect_and_translate failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                                  attempt=self.request.retries + 1, error_message=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()
```

- [ ] **Step 2: Update the pipeline chain at line 394**

Insert `detect_and_translate.si(document_id, run_id)` after `prepare_document`:

```python
pipeline = chain(
    prepare_document.si(document_id, run_id),
    detect_and_translate.si(document_id, run_id),  # NEW
    chord(
        group(
            derive_document_metadata.si(document_id, run_id),
            purge_document_derivations.si(document_id, run_id),
        ),
        derive_picture_descriptions.si(document_id, run_id),
    ),
    # ... rest unchanged
)
```

- [ ] **Step 3: Update REQUIRED_STAGES at line 2645**

Add `"detect_and_translate"` to the set.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/ -v --timeout=30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat: add detect_and_translate pipeline task with per-element language detection"
```

---

### Task 4: Update `derive_document_metadata` to prefer translated markdown

**Files:**
- Modify: `app/workers/pipeline.py:1048-1054` (markdown loading in `derive_document_metadata`)

- [ ] **Step 1: Update markdown loading**

Replace the single markdown load at lines 1048-1054 with the split approach from the spec:

```python
        # Load markdown from MinIO — use translated for summary/date/source, original for classification
        from app.services.storage import download_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived

        try:
            original_md = download_bytes_sync(bucket, f"{base_key}/docling_document.md").decode("utf-8")
        except Exception:
            logger.info("derive_document_metadata: no markdown available for %s, skipping", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_document_metadata", "COMPLETE",
                                  attempt=self.request.retries + 1,
                                  metrics={"skipped": True, "reason": "no_markdown"})
                db.commit()
            return {"stage": "derive_document_metadata", "status": "skipped", "reason": "no_markdown"}

        try:
            translated_md = download_bytes_sync(bucket, f"{base_key}/docling_document_translated.md").decode("utf-8")
        except Exception:
            translated_md = None

        # Use translated for summary/date/source (so results are English);
        # use original for classification (markings like СЕКРЕТНО should be detected in original form)
        markdown = translated_md or original_md
        classification_markdown = original_md
```

Then update the `extract_document_metadata` call to pass `classification_markdown` separately. This requires a small change to `extract_document_metadata` in `document_analysis.py` to accept an optional `classification_text` parameter — if provided, use it for the classification prompt instead of the main markdown.

- [ ] **Step 2: Update `extract_document_metadata` in `app/services/document_analysis.py`**

Add an optional `classification_text` parameter:

```python
def extract_document_metadata(markdown: str, classification_text: str | None = None) -> dict:
```

In the prompts dict, use `classification_text or markdown` for the classification call:

```python
    # Separate text for classification (may be original non-English for marking detection)
    class_text = classification_text or doc_text

    # Run prompts - classification uses class_text, others use doc_text
    # ... adjust the ThreadPoolExecutor to pass class_text for classification
```

- [ ] **Step 3: Update the call in `derive_document_metadata`**

```python
        metadata = extract_document_metadata(markdown, classification_text=classification_markdown)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/ -v --timeout=30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py app/services/document_analysis.py
git commit -m "feat: derive_document_metadata prefers translated markdown, original for classification"
```

---

### Task 5: Update cleanup paths

**Files:**
- Modify: `app/api/v1/sources.py:529` (`_hard_delete_document` MinIO cleanup)
- Modify: `app/workers/pipeline.py:1276` (`purge_document_derivations`)

- [ ] **Step 1: Add `docling_document_translated.md` to `_hard_delete_document`**

At line 529 in `sources.py`, add the translated file to the cleanup list:

```python
        for suffix in ("docling_document.md", "docling_document.json", "docling_document_translated.md"):
```

- [ ] **Step 2: Add MinIO cleanup to `purge_document_derivations`**

After the Neo4j cleanup in `purge_document_derivations`, add:

```python
        # 4. Delete translated markdown from MinIO (if exists)
        try:
            from app.services.storage import delete_object_sync
            base_key = f"artifacts/{document_id}"
            delete_object_sync(settings.minio_bucket_derived, f"{base_key}/docling_document_translated.md")
        except Exception:
            pass  # File may not exist
```

Check if `delete_object_sync` exists. If not, use the same boto3 client pattern from `storage.py`.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/pytest tests/unit/ -v --timeout=30`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/api/v1/sources.py app/workers/pipeline.py
git commit -m "fix: clean up translated markdown on document delete and purge"
```

---

## Chunk 3: API Endpoint + Frontend

### Task 6: Add translation API endpoint

**Files:**
- Modify: `app/api/v1/sources.py` (add new endpoint)
- Modify: `frontend/src/api/client.ts` (add API function)

- [ ] **Step 1: Add endpoint to `sources.py`**

After the existing metadata endpoint, add:

```python
@router.get("/documents/{document_id}/translation")
async def get_document_translation(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Return the translated markdown for a document."""
    from app.services.storage import download_bytes_async
    settings = get_settings()

    # Check document exists and has translation
    doc = (await db.execute(
        select(Document).where(Document.id == document_id)
    )).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    meta = doc.document_metadata or {}
    if not meta.get("has_translation"):
        raise HTTPException(404, "No translation available for this document")

    base_key = f"artifacts/{document_id}"
    try:
        md_bytes = await download_bytes_async(
            settings.minio_bucket_derived,
            f"{base_key}/docling_document_translated.md",
        )
        translated_md = md_bytes.decode("utf-8")
    except Exception:
        raise HTTPException(404, "Translation file not found")

    return {
        "document_id": str(document_id),
        "detected_language": meta.get("detected_language", "unknown"),
        "translated_markdown": translated_md,
    }
```

Note: Check if `download_bytes_async` exists in `storage.py`. If only the sync version exists, use `run_in_executor` to call `download_bytes_sync`.

- [ ] **Step 2: Add API function to `frontend/src/api/client.ts`**

```typescript
export interface DocumentTranslation {
  document_id: string;
  detected_language: string;
  translated_markdown: string;
}

export async function getDocumentTranslation(documentId: string): Promise<DocumentTranslation | null> {
  try {
    const resp = await fetch(`${API_BASE}/v1/documents/${documentId}/translation`);
    if (!resp.ok) return null;
    return resp.json();
  } catch {
    return null;
  }
}
```

- [ ] **Step 3: Run backend tests**

Run: `.venv/bin/pytest tests/unit/ -v --timeout=30`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/api/v1/sources.py frontend/src/api/client.ts
git commit -m "feat: add GET /documents/{id}/translation endpoint and frontend client"
```

---

### Task 7: Add translation toggle to DoclingViewer

**Files:**
- Modify: `frontend/src/components/DoclingViewer.tsx`

- [ ] **Step 1: Add state and fetch logic**

Add to the component:
- New view mode: `type ViewMode = "document" | "translate" | "json";`
- New state: `const [translatedMd, setTranslatedMd] = useState<string | null>(null);`
- Fetch translation on first toggle (lazy load, cache in state)

- [ ] **Step 2: Add "Translate" button to mode toggle**

Only show when `metadata?.has_translation` is true:

```tsx
{metadata?.has_translation && (
  <button
    className={`mode-btn${mode === "translate" ? " active" : ""}`}
    onClick={() => {
      setMode("translate");
      if (!translatedMd) {
        getDocumentTranslation(documentId).then((t) => {
          if (t) setTranslatedMd(t.translated_markdown);
        });
      }
    }}
  >
    Translate
  </button>
)}
```

- [ ] **Step 3: Add translation view rendering**

After the document iframe and before the JSON view:

```tsx
{mode === "translate" && (
  <div style={{ padding: "1rem" }}>
    <div style={{
      background: "var(--color-warning-bg, #fff3cd)",
      border: "1px solid var(--color-warning-border, #ffc107)",
      borderRadius: "var(--radius)",
      padding: "0.5rem 0.75rem",
      marginBottom: "0.75rem",
      fontSize: "0.85rem",
    }}>
      Machine-translated from {metadata?.detected_language || "unknown language"}.
      Original may contain untranslated technical terms.
    </div>
    {translatedMd ? (
      <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", fontSize: "0.9rem", lineHeight: 1.6, margin: 0 }}>
        {translatedMd}
      </pre>
    ) : (
      <div className="empty-state">
        <span className="spinner" />
        <p className="mt-sm">Loading translation...</p>
      </div>
    )}
  </div>
)}
```

- [ ] **Step 4: Build frontend**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/DoclingViewer.tsx frontend/src/api/client.ts
git commit -m "feat: add Translate toggle to DoclingViewer for foreign-language documents"
```

---

## Chunk 4: Final Verification

### Task 8: Run full test suite and verify

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/pytest tests/unit/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `ruff check app/services/translation.py app/workers/pipeline.py app/api/v1/sources.py app/services/document_analysis.py`
Expected: No errors

- [ ] **Step 3: Final commit if any lint fixes needed**

```bash
git add -A && git commit -m "chore: lint fixes"
```
