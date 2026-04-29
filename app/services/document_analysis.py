"""LLM-based document metadata extraction and picture description enrichment.

Calls Ollama for:
1. Document metadata (summary, date, classification, source) via configurable model
2. Picture descriptions via configurable multimodal model with summary context
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from app.config import get_settings

logger = logging.getLogger(__name__)


def extract_document_metadata(markdown: str, classification_text: str | None = None) -> dict:
    """Extract metadata from document markdown via LLM.

    Runs four prompts in parallel (summary, date, source, classification)
    since they're all independent reads of the same markdown.

    If ``classification_text`` is provided it is used for the classification
    prompt instead of ``markdown``.  This lets callers pass the original
    (non-translated) text for classification while using a translated version
    for the other metadata fields.
    """
    settings = get_settings()
    from app.services.ollama_clients import get_llm_client
    client = get_llm_client()
    model = settings.doc_analysis_llm_model
    think = settings.get_doc_analysis_llm_think()
    timeout = settings.doc_analysis_timeout

    max_chars = settings.ollama_num_ctx * 3
    doc_text = markdown[:max_chars] if len(markdown) > max_chars else markdown
    raw_class_text = classification_text if classification_text is not None else markdown
    class_text = raw_class_text[:max_chars] if len(raw_class_text) > max_chars else raw_class_text

    def _llm_call(system_prompt: str, user_text: str) -> str:
        return client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model=model,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
            think=think,
            timeout_s=float(timeout),  # role-specific: doc_analysis_timeout
        )

    results: dict[str, str] = {}
    non_class_prompts = {
        "document_summary": settings.doc_analysis_summary_prompt,
        "date_of_information": settings.doc_analysis_date_prompt,
        "source_characterization": settings.doc_analysis_source_prompt,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict = {
            pool.submit(_llm_call, prompt, doc_text): key
            for key, prompt in non_class_prompts.items()
        }
        # Classification runs against the original (un-translated) text
        futures[pool.submit(_llm_call, settings.doc_analysis_classification_prompt, class_text)] = "classification"

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                logger.info("Document metadata '%s' extracted", key)
            except Exception as e:
                logger.warning("Document metadata '%s' failed: %s", key, e)
                results[key] = "Unknown" if key != "classification" else "UNCLASSIFIED"

    # Normalize classification
    valid_classes = {"UNCLASSIFIED", "CUI", "FOUO", "SECRET", "TOP SECRET"}
    classification = results.get("classification", "UNCLASSIFIED").upper().strip()
    if classification not in valid_classes:
        classification = "UNCLASSIFIED"

    return {
        "document_summary": results.get("document_summary", ""),
        "date_of_information": results.get("date_of_information", "Unknown"),
        "classification": classification,
        "source_characterization": results.get("source_characterization", "Unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def describe_pictures(docling_json: dict, document_summary: str) -> dict:
    """Enrich picture items in Docling JSON with LLM-generated descriptions.

    Iterates PictureItems with embedded images, sends them to the configured
    multimodal model with the document summary as context. Uses bounded
    parallelism (max 3 concurrent) to avoid overwhelming Ollama.

    Returns the modified docling_json dict.
    """
    settings = get_settings()
    model = settings.picture_description_model
    timeout = settings.picture_description_timeout
    prompt_template = settings.picture_description_prompt.replace("\\n", "\n")
    prompt = prompt_template.replace("{document_summary}", document_summary)

    # Collect pictures that have image data
    pictures = docling_json.get("pictures", [])
    if not isinstance(pictures, list):
        return docling_json

    describable: list[tuple[int, str]] = []  # (index, base64)
    for idx, pic in enumerate(pictures):
        if not isinstance(pic, dict):
            continue
        image_ref = pic.get("image", {})
        uri = image_ref.get("uri", "") if isinstance(image_ref, dict) else ""
        if uri and uri.startswith("data:") and "," in uri:
            b64 = uri.split(",", 1)[1]
            if b64:
                describable.append((idx, b64))

    if not describable:
        logger.info("Picture descriptions: 0 describable pictures found")
        return docling_json

    max_workers = min(settings.picture_desc_concurrency, len(describable))
    descriptions: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_describe_single_image, b64, prompt, model, timeout, settings): idx
            for idx, b64 in describable
        }
        for future in as_completed(futures):
            pic_idx = futures[future]
            try:
                desc = future.result()
                if desc:
                    descriptions[pic_idx] = desc
            except Exception as e:
                logger.warning("Picture %d description failed: %s", pic_idx, e)

    # Apply descriptions to the Docling JSON
    for idx, desc in descriptions.items():
        pic = pictures[idx]
        pic["description"] = desc
        if "annotations" not in pic:
            pic["annotations"] = []
        pic["annotations"].append({
            "kind": "description",
            "text": desc,
            "source": "llm",
            "model": model,
        })

    logger.info(
        "Picture descriptions: found=%d, described=%d, model=%s",
        len(describable), len(descriptions), model,
    )
    return docling_json


def _describe_single_image(
    image_b64: str, prompt: str, model: str, timeout: int, settings
) -> str | None:
    """Send a single image to the multimodal LLM for description."""
    try:
        from app.services.ollama_clients import get_vlm_client
        client = get_vlm_client()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]
        content = client.chat(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
            think=settings.get_picture_description_think(),
            timeout_s=float(timeout),  # picture_description_timeout per call
        )
        logger.debug("Picture description (%d chars): %.100s...", len(content) if content else 0, content or "")
        return content
    except Exception as e:
        logger.warning("Picture description failed: %s", e)
        return None
