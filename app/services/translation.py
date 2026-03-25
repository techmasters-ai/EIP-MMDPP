"""Foreign language detection and LLM-based translation.

Detects non-English content per-element using langdetect, then translates
flagged elements via Ollama in batches with boundary markers.
"""

import logging
import re
from collections import Counter

import httpx
from langdetect import detect_langs, LangDetectException
from langdetect.detector_factory import DetectorFactory

from app.config import get_settings

logger = logging.getLogger(__name__)

# Deterministic detection
DetectorFactory.seed = 0

_BATCH_CHAR_LIMIT = 2000

# Non-Latin scripts: Cyrillic, CJK, Arabic, Devanagari, Thai, Hebrew, Korean
_NON_LATIN = re.compile(
    r'[\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff\u0600-\u06ff'
    r'\u0900-\u097f\u0e00-\u0e7f\u0590-\u05ff\uac00-\ud7af]'
)
_BOUNDARY = "\n---ELEMENT_BOUNDARY---\n"
_BOUNDARY_STRIPPED = "---ELEMENT_BOUNDARY---"


def _detect_non_english(text: str, min_length: int) -> str | None:
    """Return the detected language code if *text* is non-English, else None."""
    if len(text) < min_length:
        return None
    try:
        langs = detect_langs(text)
        top = langs[0]
        if top.lang != "en":
            return top.lang
    except LangDetectException:
        pass
    return None


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
    settings = get_settings()
    min_detect_length = settings.translation_min_detect_length

    non_english: list[int] = []
    lang_counts: Counter = Counter()

    for i, elem in enumerate(elements):
        text = elem.get("content_text", "") or ""
        if len(text) < min_detect_length:
            continue

        # If text contains non-Latin characters, flag for translation directly —
        # langdetect misclassifies mixed-language text as English when English dominates.
        if _NON_LATIN.search(text):
            non_english.append(i)
            # Try to identify the non-English language for doc-level stats
            detected_lang = _detect_non_english(text, min_detect_length)
            lang_counts[detected_lang or "unknown"] += 1
            continue

        detected_lang = _detect_non_english(text, min_detect_length)
        if detected_lang:
            non_english.append(i)
            lang_counts[detected_lang] += 1

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

    # Resolve settings and create client once for all batches
    settings = get_settings()
    url = f"{settings.get_ollama_llm_url()}/v1/chat/completions"
    model = settings.translation_model
    prompt = settings.translation_prompt.replace("\\n", "\n")
    timeout = settings.translation_timeout
    max_tokens = settings.llm_max_tokens

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

    logger.info("translate_elements: %d batches for %d non-English elements",
                len(batches), len(non_english_indices))

    # Single client for all translation calls (connection reuse)
    with httpx.Client(timeout=timeout) as client:
        for batch_num, batch_indices in enumerate(batches):
            if len(batch_indices) == 1:
                idx = batch_indices[0]
                translated = _ollama_translate(
                    client, url, model, prompt, elements[idx]["content_text"],
                    timeout=timeout, max_tokens=max_tokens,
                )
                logger.info("translate batch %d (single idx=%d): input=%d chars, output=%d chars, changed=%s",
                            batch_num, idx, len(elements[idx]["content_text"]),
                            len(translated) if translated else 0,
                            bool(translated and translated.strip() != elements[idx]["content_text"]))
                if translated:
                    result[idx] = translated.strip()
            else:
                combined = _BOUNDARY.join(elements[idx]["content_text"] for idx in batch_indices)
                translated = _ollama_translate(
                    client, url, model, prompt, combined,
                    timeout=timeout, max_tokens=max_tokens,
                )
                logger.info("translate batch %d (%d elements): input=%d chars, output=%d chars, has_boundary=%s",
                            batch_num, len(batch_indices), len(combined),
                            len(translated) if translated else 0,
                            bool(translated and _BOUNDARY_STRIPPED in translated))

                if translated and _BOUNDARY_STRIPPED in translated:
                    parts = translated.split(_BOUNDARY_STRIPPED)
                    if len(parts) == len(batch_indices):
                        for idx, part in zip(batch_indices, parts):
                            result[idx] = part.strip()
                    else:
                        logger.info("translate batch %d: boundary count mismatch (got %d, expected %d), falling back",
                                    batch_num, len(parts), len(batch_indices))
                        _translate_individually(
                            client, url, model, prompt, elements, batch_indices, result,
                            timeout=timeout, max_tokens=max_tokens,
                        )
                else:
                    _translate_individually(
                        client, url, model, prompt, elements, batch_indices, result,
                        timeout=timeout, max_tokens=max_tokens,
                    )

    return result


def _translate_individually(
    client: httpx.Client, url: str, model: str, prompt: str,
    elements: list[dict], indices: list[int], result: list[str],
    *, timeout: float, max_tokens: int,
) -> None:
    """Fallback: translate each element individually."""
    for idx in indices:
        translated = _ollama_translate(
            client, url, model, prompt, elements[idx]["content_text"],
            timeout=timeout, max_tokens=max_tokens,
        )
        if translated:
            result[idx] = translated.strip()


def _ollama_translate(
    client: httpx.Client, url: str, model: str, prompt: str, text: str,
    *, timeout: float, max_tokens: int,
) -> str | None:
    """Send text to Ollama for translation using shared client."""
    from app.services.document_analysis import _ollama_chat

    try:
        return _ollama_chat(
            client, url, model,
            [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return None
