"""Worker-side port of docling-graph's _sanitize_docling_document.

Lives in the worker codebase so build_extraction_index_hybrid can apply
sanitization BEFORE the HybridChunker runs. The merged chunks written to
ArcadeDB then have already-clean text — Phase 2 chunked-mode benefits even
though docling-graph's own sanitize step is skipped.

Functionally byte-identical to docling-graph's main.py:437 version
(commit 2026-05-01). Keep both copies in sync — if you change the rule
set here, mirror the change in docker/docling-graph/app/main.py and vice
versa.
"""
from __future__ import annotations

import re

_AD_TRACKING_DOMAINS = (
    "adroll.com",
    "adrta.com",
    "doubleclick.net",
    "googletagmanager.com",
    "googletagservices.com",
    "googleadservices.com",
    "google-analytics.com",
    "googlesyndication.com",
    "facebook.com/tr",
    "amazon-adsystem.com",
    "adservice.google",
    "scorecardresearch.com",
)

# Standalone URL or markdown link with no surrounding prose. Captures patterns
# like "[Foo](https://...)" or just "https://example.com/path" with optional
# surrounding whitespace. Leading list markers ("- ", "* ") are tolerated so
# entire navigation/sidebar lists collapse cleanly.
_PURE_LINK_LINE = re.compile(
    r"^\s*[-*]?\s*"
    r"(?:\[[^\]]*\]\([^)]+\)|https?://\S+|<https?://\S+>)"
    r"\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

# Long unbroken token in the base64 alphabet (incl. URL-safe variants).
_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{64,}={0,2}")

# Single %XX triplet. Counted per whitespace-delimited token so a sentence
# with an occasional encoded URL doesn't false-trigger.
_PERCENT_TRIPLET = re.compile(r"%[0-9A-Fa-f]{2}")


def _contains_encoded_blob(text: str) -> bool:
    """True if any whitespace-delimited token looks like an obfuscated blob.

    (a) base64 — long token with padding or mixed-case+digit composition
    (b) percent-encoded URL fragment — long token with >=6 %XX triplets
    """
    if not isinstance(text, str) or not text:
        return False
    for m in _BASE64_TOKEN.finditer(text):
        tok = m.group(0)
        has_padding = ("=" in tok) or ("+" in tok) or ("/" in tok)
        has_mixed = (
            any(c.isupper() for c in tok)
            and any(c.islower() for c in tok)
            and any(c.isdigit() for c in tok)
        )
        if has_padding or has_mixed:
            return True
    for tok in text.split():
        if len(tok) < 64:
            continue
        if len(_PERCENT_TRIPLET.findall(tok)) >= 6:
            return True
    return False


def _looks_like_nav_or_tracking(text: str) -> bool:
    """Return True if the entire text is web cruft to drop pre-extraction.

    Rules (conservative — false negatives preferred over false positives):
      1. Text contains an ad-tracking domain.
      2. Text is ONLY pure markdown links / bare URLs (sidebar nav, link lists).
      3. Text contains an obfuscated/encoded blob.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower()
    for dom in _AD_TRACKING_DOMAINS:
        if dom in lowered:
            return True

    nonblank = [ln for ln in text.splitlines() if ln.strip()]
    if not nonblank:
        return False
    if all(_PURE_LINK_LINE.match(ln) for ln in nonblank):
        return True

    if _contains_encoded_blob(text):
        return True

    return False


def sanitize_docling_document(doc: dict, stats: dict | None = None) -> dict:
    """Blank text content of ad-tracking / navigation-list elements.

    Mirrors docling-graph's _sanitize_docling_document (main.py:437) — does
    NOT remove elements from texts[], REPLACES their `text` / `orig` with
    empty strings AND clears the `hyperlink` annotation. Element + $refs
    stay in place; HybridChunker treats empty/whitespace texts as
    zero-token contributions so they vanish from the markdown the LLM sees.

    Protects label=="caption" unconditionally (image-description prose).

    Tables (doc["tables"][]) are NEVER touched — only doc["texts"][] is
    walked. Table grids and per-cell structure are fully preserved.

    Args:
      doc: DoclingDocument JSON dict.
      stats: optional dict mutated with ``texts_in`` and ``texts_dropped``
        counters (kept named ``texts_dropped`` for compatibility with the
        docling-graph original; semantically "blanked", not removed).

    Returns:
      Same doc with sanitized texts[] (new dict only when changes occurred;
      original returned unchanged otherwise to avoid the copy).
    """
    if stats is None:
        stats = {}

    texts_in = list(doc.get("texts") or [])
    stats["texts_in"] = len(texts_in)

    new_texts: list = []
    blanked = 0
    for t in texts_in:
        if not isinstance(t, dict):
            new_texts.append(t)
            continue
        label = (t.get("label") or "").lower()
        if label == "caption":
            new_texts.append(t)
            continue
        text_str = t.get("text") or t.get("orig") or ""
        hyperlink = t.get("hyperlink") or ""
        # Render the item the way the chunker's format_batch_markdown will:
        # `[text](hyperlink)` when hyperlinked, bare text otherwise.
        if hyperlink:
            rendered = f"[{text_str}]({hyperlink})"
        else:
            rendered = text_str
        if _looks_like_nav_or_tracking(rendered):
            blanked_t = dict(t)
            blanked_t["text"] = ""
            blanked_t["orig"] = ""
            if "hyperlink" in blanked_t:
                blanked_t["hyperlink"] = None
            new_texts.append(blanked_t)
            blanked += 1
            continue
        new_texts.append(t)

    stats["texts_dropped"] = blanked

    if blanked == 0:
        return doc

    new_doc = dict(doc)
    new_doc["texts"] = new_texts
    return new_doc
