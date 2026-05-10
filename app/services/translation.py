"""Foreign language detection and LLM-based translation.

Detects non-English content per-element using langdetect, then translates
flagged elements via Ollama in batches with boundary markers.
"""

import logging
import re
from collections import Counter

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
            "document_language": "ru",           # most common non-English, or "en"
            "non_english_indices": [0, 2, 5],    # indices needing translation
            "language_confidences": {"ru": 2, "unknown": 4},  # per-language counts;
                                                 # "unknown" = regex-flagged but
                                                 # langdetect couldn't classify
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

        # Flag elements containing non-Latin characters (Cyrillic, CJK, Arabic, etc.)
        # Only these need translation — langdetect on short English text produces
        # too many false positives that dilute batches and confuse the LLM.
        if _NON_LATIN.search(text):
            non_english.append(i)
            detected_lang = _detect_non_english(text, min_detect_length)
            lang_counts[detected_lang or "unknown"] += 1

    doc_lang = lang_counts.most_common(1)[0][0] if lang_counts else "en"
    return {
        "document_language": doc_lang,
        "non_english_indices": non_english,
        "language_confidences": dict(lang_counts),
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

    # Resolve settings once for all batches
    settings = get_settings()
    model = settings.translation_model
    think = settings.get_translation_think()
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

    for batch_num, batch_indices in enumerate(batches):
        if len(batch_indices) == 1:
            idx = batch_indices[0]
            translated = _ollama_translate(
                model, prompt, elements[idx]["content_text"],
                temperature=0.1, max_tokens=max_tokens, timeout=timeout, think=think,
            )
            input_chars = len(elements[idx]["content_text"])
            output_chars = len(translated) if translated else 0
            logger.info("translate batch %d (single idx=%d): input=%d chars, output=%d chars, changed=%s",
                        batch_num, idx, input_chars, output_chars,
                        bool(translated and translated.strip() != elements[idx]["content_text"]))
            if input_chars > 0 and output_chars == 0:
                logger.error(
                    "TRANSLATION_SILENT_DROP batch=%d idx=%d input_chars=%d output_chars=0 — "
                    "non-empty source but empty translation; element kept as-is, search "
                    "recall degraded.",
                    batch_num, idx, input_chars,
                )
            if translated:
                result[idx] = translated.strip()
        else:
            combined = _BOUNDARY.join(elements[idx]["content_text"] for idx in batch_indices)
            translated = _ollama_translate(
                model, prompt, combined,
                temperature=0.1, max_tokens=max_tokens, timeout=timeout, think=think,
            )
            input_chars = len(combined)
            output_chars = len(translated) if translated else 0
            logger.info("translate batch %d (%d elements): input=%d chars, output=%d chars, has_boundary=%s",
                        batch_num, len(batch_indices), input_chars, output_chars,
                        bool(translated and _BOUNDARY_STRIPPED in translated))
            if input_chars > 0 and output_chars == 0:
                # Recall recovery: when the batched call returned literally
                # nothing (rare — model refused, hit early stop, or both
                # `content` and `reasoning_content` were empty), the OLD
                # behavior was to log ERROR and drop. Fall through to
                # individual per-element translation instead, matching the
                # other failure modes (no-boundary / wrong-boundary count).
                logger.error(
                    "TRANSLATION_SILENT_DROP batch=%d batch_size=%d input_chars=%d "
                    "output_chars=0 — empty batched response; falling back to "
                    "individual per-element translation for %d elements.",
                    batch_num, len(batch_indices), input_chars, len(batch_indices),
                )
                _translate_individually(
                    model, prompt, elements, batch_indices, result,
                    timeout=timeout, max_tokens=max_tokens, think=think,
                )
            elif translated and _BOUNDARY_STRIPPED in translated:
                parts = translated.split(_BOUNDARY_STRIPPED)
                if len(parts) == len(batch_indices):
                    for idx, part in zip(batch_indices, parts):
                        result[idx] = part.strip()
                else:
                    logger.info("translate batch %d: boundary count mismatch (got %d, expected %d), falling back",
                                batch_num, len(parts), len(batch_indices))
                    _translate_individually(
                        model, prompt, elements, batch_indices, result,
                        timeout=timeout, max_tokens=max_tokens, think=think,
                    )
            else:
                _translate_individually(
                    model, prompt, elements, batch_indices, result,
                    timeout=timeout, max_tokens=max_tokens, think=think,
                )

    return result


def _translate_individually(
    model: str, prompt: str,
    elements: list[dict], indices: list[int], result: list[str],
    *, timeout: float, max_tokens: int, think: str | bool | None,
) -> None:
    """Fallback: translate each element individually."""
    for idx in indices:
        translated = _ollama_translate(
            model, prompt, elements[idx]["content_text"],
            temperature=0.1, max_tokens=max_tokens, timeout=timeout, think=think,
        )
        if translated:
            result[idx] = translated.strip()


def _ollama_translate(
    model: str, prompt: str, text: str,
    *, temperature: float = 0.1, max_tokens: int, timeout: int,
    think: str | bool | None = None,
) -> str | None:
    """Single-element translation via pool client."""
    from app.services.ollama_clients import get_translation_client

    try:
        pool_client = get_translation_client()
        return pool_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            timeout_s=float(timeout),  # role-specific: translation_timeout
        )
    except Exception as e:
        # ERROR not WARNING — every silent translation drop degrades downstream
        # search recall (translated text is the bridge between non-English
        # source and English-trained embeddings). Surface enough context that
        # an operator can find the offending element + retry strategy.
        logger.error(
            "TRANSLATION_FAILED model=%s timeout_s=%s exc_type=%s exc_msg=%s "
            "input_chars=%d — element will be kept as original-language text; "
            "downstream embedding/search will be degraded for this batch.",
            model, timeout, type(e).__name__, str(e), len(text),
        )
        return None
