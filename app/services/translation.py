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
            idx = batch_indices[0]
            translated = _ollama_translate(elements[idx]["content_text"])
            if translated:
                result[idx] = translated.strip()
        else:
            combined = _BOUNDARY.join(elements[idx]["content_text"] for idx in batch_indices)
            translated = _ollama_translate(combined)

            if translated and _BOUNDARY.strip() in translated:
                parts = translated.split(_BOUNDARY.strip())
                if len(parts) == len(batch_indices):
                    for idx, part in zip(batch_indices, parts):
                        result[idx] = part.strip()
                else:
                    _translate_individually(elements, batch_indices, result)
            else:
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
    from app.services.document_analysis import _ollama_chat

    settings = get_settings()
    url = f"{settings.ollama_base_url}/v1/chat/completions"
    prompt = settings.translation_prompt.replace("\\n", "\n")

    try:
        with httpx.Client(timeout=settings.translation_timeout) as client:
            return _ollama_chat(
                client, url, settings.translation_model,
                [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                temperature=0.1,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.translation_timeout,
            )
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return None
