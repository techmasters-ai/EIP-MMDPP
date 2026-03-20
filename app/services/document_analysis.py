"""LLM-based document metadata extraction and picture description enrichment.

Calls Ollama for:
1. Document metadata (summary, date, classification, source) via configurable model
2. Picture descriptions via configurable multimodal model with summary context
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def _ollama_chat(
    client: httpx.Client,
    url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int,
    timeout: float = 300,
) -> str:
    """Shared Ollama chat completion call. Returns stripped assistant content."""
    resp = client.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extract_document_metadata(markdown: str) -> dict:
    """Extract metadata from document markdown via LLM.

    Runs four prompts in parallel (summary, date, source, classification)
    since they're all independent reads of the same markdown.
    """
    settings = get_settings()
    model = settings.doc_analysis_llm_model
    timeout = settings.doc_analysis_timeout
    url = f"{settings.ollama_base_url}/v1/chat/completions"

    # Shared client for connection reuse across parallel calls (httpx.Client is thread-safe)
    client = httpx.Client(timeout=timeout)

    max_tokens = settings.llm_max_tokens

    def _llm_call(system_prompt: str, user_text: str) -> str:
        return _ollama_chat(
            client, url, model,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            temperature=0.1, max_tokens=max_tokens, timeout=timeout,
        )

    # Truncate markdown to avoid exceeding context window
    max_chars = settings.ollama_num_ctx * 3
    doc_text = markdown[:max_chars] if len(markdown) > max_chars else markdown

    # Run all 4 prompts in parallel — they're independent
    results: dict[str, str] = {}
    prompts = {
        "document_summary": settings.doc_analysis_summary_prompt,
        "date_of_information": settings.doc_analysis_date_prompt,
        "source_characterization": settings.doc_analysis_source_prompt,
        "classification": settings.doc_analysis_classification_prompt,
    }

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_llm_call, prompt, doc_text): key
                for key, prompt in prompts.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                    logger.info("Document metadata '%s' extracted", key)
                except Exception as e:
                    logger.warning("Document metadata '%s' failed: %s", key, e)
                    results[key] = "Unknown" if key != "classification" else "UNCLASSIFIED"
    finally:
        client.close()

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

    # Parallel description with bounded concurrency (max 3 to avoid Ollama overload)
    max_workers = min(3, len(describable))
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
        url = f"{settings.ollama_base_url}/v1/chat/completions"
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
        # Use a per-call client (images are large payloads; keep-alive less beneficial)
        with httpx.Client(timeout=timeout) as client:
            content = _ollama_chat(
                client, url, model, messages,
                temperature=0.2, max_tokens=settings.llm_max_tokens, timeout=timeout,
            )
        logger.debug("Picture description (%d chars): %.100s...", len(content), content)
        return content
    except Exception as e:
        logger.warning("Picture description failed: %s", e)
        return None
