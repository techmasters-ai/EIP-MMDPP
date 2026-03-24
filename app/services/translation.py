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

_MIN_DETECT_LENGTH = 50
_MIN_DETECT_LENGTH_CJK = 5
_BATCH_CHAR_LIMIT = 2000
_BOUNDARY = "\n---ELEMENT_BOUNDARY---\n"
_BOUNDARY_STRIPPED = "---ELEMENT_BOUNDARY---"

# CJK Unicode ranges: Chinese, Japanese, Korean characters
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')


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
        min_len = _MIN_DETECT_LENGTH_CJK if _CJK_RANGE.search(text) else _MIN_DETECT_LENGTH
        if len(text) < min_len:
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

    # Resolve settings and create client once for all batches
    settings = get_settings()
    url = f"{settings.ollama_base_url}/v1/chat/completions"
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

    # Single client for all translation calls (connection reuse)
    with httpx.Client(timeout=timeout) as client:
        for batch_indices in batches:
            if len(batch_indices) == 1:
                idx = batch_indices[0]
                translated = _ollama_translate(
                    client, url, model, prompt, elements[idx]["content_text"],
                    timeout=timeout, max_tokens=max_tokens,
                )
                if translated:
                    result[idx] = translated.strip()
            else:
                combined = _BOUNDARY.join(elements[idx]["content_text"] for idx in batch_indices)
                translated = _ollama_translate(
                    client, url, model, prompt, combined,
                    timeout=timeout, max_tokens=max_tokens,
                )

                if translated and _BOUNDARY_STRIPPED in translated:
                    parts = translated.split(_BOUNDARY_STRIPPED)
                    if len(parts) == len(batch_indices):
                        for idx, part in zip(batch_indices, parts):
                            result[idx] = part.strip()
                    else:
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
